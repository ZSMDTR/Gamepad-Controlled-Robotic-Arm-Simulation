"""Loopback UI and best-effort MuJoCo physics with 100 Hz pose control.

The browser only supplies intent. Network cadence does not set the solver dt.
This module has no robot driver or hardware command path.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import mujoco
import numpy as np

from .mujoco_backend import MujocoBackend, available_models
from .pose_control import PoseController
from .recorder import capture_state

DT = 0.01
INPUT_TIMEOUT = 0.25
LEADER_TIMEOUT = 0.5
STATIC = Path(__file__).parent / "static"


def physics_steps(timestep):
    """Number of physics steps per control cycle, without changing model time."""
    count = round(DT / timestep)
    if count < 1 or not math.isclose(count * timestep, DT, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("model timestep must divide the 0.01 s control period")
    return count


def input_vector(value):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("direction must contain three numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) or abs(x) > 1 for x in value):
        raise ValueError("direction values must be finite numbers in [-1, 1]")
    return [float(x) for x in value]


def leader_configuration(config):
    """Validate the user-editable experimental mapping before connecting."""
    if not isinstance(config, dict) or set(config) != {'joint_indices', 'joint_scales', 'gripper_scale'}:
        raise ValueError('leader config requires joint_indices, joint_scales and gripper_scale')
    indices = config['joint_indices']
    scales = config['joint_scales']
    grip = config['gripper_scale']
    if (not isinstance(indices, list) or len(indices) != 6 or
            any(type(x) is not int for x in indices) or set(indices) != set(range(6))):
        raise ValueError('joint_indices must be a permutation of 0..5')
    if (not isinstance(scales, list) or len(scales) != 6 or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                not math.isfinite(x) or abs(x) > 2 for x in scales) or
            isinstance(grip, bool) or not isinstance(grip, (int, float)) or
            not math.isfinite(grip) or abs(grip) > 2):
        raise ValueError('joint_scales and gripper_scale must be finite numbers in [-2, 2]')
    return {'joint_indices': list(indices), 'joint_scales': list(scales), 'gripper_scale': grip}


class Simulation:
    """One physics owner thread; commands and snapshots are protected by a lock."""
    def __init__(self):
        self.lock = threading.Lock()
        self.native_hold_on_release = False
        self._native_previous_rotation = np.zeros(3)
        self.recorder = None
        self.recording_error = None
        self.leader_active = False
        self.leader_sample = None
        self.leader_config = None
        self.leader_origin = None
        self.leader_unwrapped = None
        self.leader_read_error = None
        self.leader_q_origin = None
        self.leader_ref = None
        self.leader_velocity = None
        self.leader_gripper_origin = None
        self.leader_status = 'OFF'
        self.stop_event = threading.Event()
        self.method = "dls"
        self.elbow_weight = .35
        self.backend = MujocoBackend("ur5e")
        self.physics_steps = physics_steps(self.backend.timestep)
        self.controller = self._new_controller()
        self.direction = [0.0] * 3
        self.rotation_input = [0.0] * 3
        self.gripper_input = 0
        self.gripper_target = self.backend.state()["gripper_position"]
        self.enabled = False
        self.needs_release = False
        self.last_input = float("-inf")
        self.tick_times = deque(maxlen=1000)
        self.tick_ms = deque(maxlen=1000)
        self.overruns = 0
        self.fault = None
        self.result = None
        self.thread = threading.Thread(target=self.run, name="mujoco-simulator", daemon=True)

    def _new_controller(self):
        spec = dict(self.backend.spec, mode=self.method, posture_gain=.2 * self.elbow_weight)
        return PoseController(spec)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)

    def set_input(self, body):
        allowed = {"direction", "rotation", "gripper", "enabled"}
        if not {"direction", "enabled"} <= set(body) or not set(body) <= allowed or type(body["enabled"]) is not bool:
            raise ValueError("expected direction, optional rotation and gripper, and boolean enabled")
        direction = input_vector(body["direction"])
        rotation = input_vector(body.get("rotation", [0, 0, 0]))
        gripper = body.get("gripper", 0)
        if type(gripper) is not int or gripper not in (-1, 0, 1):
            raise ValueError("gripper must be -1, 0 or 1")
        with self.lock:
            if not body["enabled"]:
                self.needs_release = False
            self.direction = direction
            self.rotation_input = rotation
            self.gripper_input = gripper
            self.enabled = body["enabled"] and not self.needs_release and self.fault is None
            self.last_input = time.monotonic()

    @staticmethod
    def _leader_values(sample, now):
        try:
            positions = np.asarray(sample.positions, dtype=float)
            timestamp = float(sample.timestamp)
        except (AttributeError, TypeError, ValueError):
            return None
        if (positions.shape != (7,) or not np.isfinite(positions).all()
                or not math.isfinite(timestamp) or timestamp > now + .01
                or now - timestamp > LEADER_TIMEOUT):
            return None
        return positions.copy(), timestamp

    def start_leader(self, sample, config, now):
        """Arm relative following only after a fresh read and explicit toggle."""
        values = self._leader_values(sample, now)
        config = leader_configuration(config)
        scales = np.asarray(config['joint_scales'], dtype=float)
        indices = np.asarray(config['joint_indices'])
        grip_scale = float(config['gripper_scale'])
        if len(self.backend.qpos_ids) != 6:
            raise ValueError('leader trial supports six-joint robots only')
        if values is None:
            self.leader_status = 'NO FRESH DATA'
            return False
        with self.lock:
            if self.fault is not None:
                self.leader_status = 'SIMULATION FAULT'
                return False
            s = self.backend.state()
            self.controller.reset()
            self.leader_config = {'joint_indices': indices.copy(), 'joint_scales': scales.copy(), 'gripper_scale': grip_scale}
            self.leader_origin = values[0]
            self.leader_sample = values
            self.leader_unwrapped = values[0].copy()
            self.leader_read_error = None
            self.leader_q_origin = s['q'].copy()
            self.leader_ref = s['q'].copy()
            self.leader_velocity = np.zeros(6)
            self.leader_gripper_origin = float(s['gripper_position'])
            self.gripper_target = self.leader_gripper_origin
            self.leader_active = True
            self.leader_status = 'ON - TRIAL'
        return True

    def set_leader_sample(self, sample, now):
        values = self._leader_values(sample, now)
        with self.lock:
            if values is not None and self.leader_sample is not None:
                if values[1] < self.leader_sample[1]:
                    self.leader_read_error = 'OUT OF ORDER'
                elif values[1] > self.leader_sample[1]:
                    delta = (values[0] - self.leader_sample[0] + np.pi) % (2 * np.pi) - np.pi
                    if np.any(np.abs(delta) > np.deg2rad(25)):
                        self.leader_read_error = 'JUMP'
                    else:
                        self.leader_unwrapped += delta
            self.leader_sample = values

    def stop_leader(self, reason='OFF'):
        with self.lock:
            self.leader_active = False
            self.leader_status = reason
            self.leader_sample = None
            self.leader_read_error = None
            self.controller.reset()

    def _leader_command(self, state, positions):
        """Relative joint trial with low speed, acceleration and lead limits."""
        spec = self.backend.spec
        indices = self.leader_config['joint_indices']
        desired = np.clip(self.leader_q_origin + (positions[indices] - self.leader_origin[indices]) *
                          self.leader_config['joint_scales'], spec['lower'], spec['upper'])
        vmax = np.minimum(spec['velocity_limit'], .1)
        amax = np.minimum(spec['acceleration_limit'], .5)
        wanted = np.clip((desired - self.leader_ref) / DT, -vmax, vmax)
        velocity = np.clip(wanted, self.leader_velocity - amax * DT, self.leader_velocity + amax * DT)
        ref = self.leader_ref + velocity * DT
        ref = np.clip(ref, np.maximum(spec['lower'], state['q'] - .03),
                      np.minimum(spec['upper'], state['q'] + .03))
        self.leader_velocity = (ref - self.leader_ref) / DT
        self.leader_ref = ref
        grip = self.leader_gripper_origin + (positions[6] - self.leader_origin[6]) * self.leader_config['gripper_scale']
        self.gripper_target += float(np.clip(np.clip(grip, 0., 1.) - self.gripper_target, -.2 * DT, .2 * DT))
        self.backend.command(ref, self.gripper_target)
        return {'q_ref': ref.copy(), 'constraints': ['leader_trial'], 'target_position': state['tcp'].copy(),
                'target_rotation': state['rotation'].copy(), 'error_m': 0., 'error_rad': 0.,
                'mode': 'leader_trial', 'solve_ms': 0.}

    def configure(self, body):
        if not body or not set(body) <= {"method", "elbow_weight", "robot"}:
            raise ValueError("expected method, robot and/or elbow_weight")
        if "method" in body and body["method"] not in ("dls", "sdls", "position_priority", "constrained_dls"):
            raise ValueError("unknown controller method")
        if "robot" in body and body["robot"] not in available_models():
            raise ValueError("unknown robot")
        if "elbow_weight" in body:
            x = body["elbow_weight"]
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1:
                raise ValueError("elbow_weight must be in [0, 1]")
        with self.lock:
            if "method" in body:
                self.method = body["method"]
                self.controller.set_mode(self.method)
            if "elbow_weight" in body:
                self.elbow_weight = float(body["elbow_weight"])
                self.controller.posture_gain = .2 * self.elbow_weight
            if "robot" in body and body["robot"] != self.backend.name:
                new_backend = MujocoBackend(body["robot"])
                new_steps = physics_steps(new_backend.timestep)
                self.backend = new_backend
                self.physics_steps = new_steps
                self.controller = self._new_controller()
                self._reset_input()
                self.gripper_target = self.backend.state()["gripper_position"]
                self.result = None
                self.fault = None

    def _reset_input(self):
        self.enabled = False
        self.direction = [0.0] * 3
        self.rotation_input = [0.0] * 3
        self.gripper_input = 0
        self.last_input = float("-inf")
        self.needs_release = True

    def reset(self):
        with self.lock:
            self._reset_input()
            self.fault = None
            measured = self.backend.reset()
            self.controller.reset()
            self.gripper_target = measured["gripper_position"]
            self.result = None
            self.tick_ms.clear()
            self.tick_times.clear()
            self.overruns = 0

    def snapshot(self):
        with self.lock:
            s = self.backend.state()
            r = self.result
            stale = time.monotonic() - self.last_input > INPUT_TIMEOUT
            target = s["tcp"] if r is None else r["target_position"]
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, s["rotation"].ravel())
            joint_units = ["m" if self.backend.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_SLIDE else "rad" for i in self.backend.joint_ids]
            result = dict(joints=s["q"].tolist(), joint_units=joint_units, joint_velocity=s["qvel"].tolist(),
                          points=s["points"].tolist(), tcp=s["tcp"].tolist(), target=target.tolist(),
                          tcp_rotation=s["rotation"].tolist(), tcp_quat=quat.tolist(),
                          target_rotation=(s["rotation"] if r is None else r["target_rotation"]).tolist(),
                          error_m=float(np.linalg.norm(target - s["tcp"])),
                          orientation_error_rad=0 if r is None else r["error_rad"],
                          solve_ms=0 if r is None else r["solve_ms"],
                          sigma_min=float(np.linalg.svd(s["jacobian"][:3], compute_uv=False)[-1]),
                          constraints=["hold"] if r is None else r["constraints"],
                          method=self.method, controller_mode=self.method, robot_id=self.backend.name, available_models=available_models(),
                          elbow_weight=self.elbow_weight, gripper_position=s["gripper_position"],
                          gripper_range=s["gripper_range"], gripper_target=self.gripper_target,
                          finger_positions=s["finger_positions"].tolist(), gripper_available=bool(len(self.backend.finger_ids)),
                          actuator_force=s["actuator_force"].tolist(),
                          joint_lower=s["joint_lower"].tolist(), joint_upper=s["joint_upper"].tolist(),
                          simulation_time=s["time"], input_stale=stale,
                          enabled=self.enabled and not stale and self.fault is None,
                          needs_release=self.needs_release, fault=self.fault,
                          simulation_only=True, physics="MuJoCo", nominal_hz=1 / DT,
                          joint_velocity_cmd=([0.] * len(s["q"]) if r is None else r["qdot_cmd"].tolist()),
                          tick_ms=self.tick_ms[-1] if self.tick_ms else 0, overruns=self.overruns)
            times = self.tick_times
            result["measured_hz"] = ((len(times) - 1) / (times[-1] - times[0])) if len(times) > 1 else 0
            ordered = sorted(self.tick_ms)
            result["tick_p99_ms"] = ordered[math.ceil(.99 * len(ordered)) - 1] if ordered else 0
            return result

    def tick(self, start, deadline):
        """Advance one control period. The caller supplies any viewer lock."""
        with self.lock:
            leader_mode = self.leader_active
            leader_sample = self.leader_sample
            leader_fresh = (leader_sample is not None and self.leader_read_error is None and
                            0 <= start - leader_sample[1] <= LEADER_TIMEOUT)
            if leader_mode and (not leader_fresh or start - deadline > INPUT_TIMEOUT):
                self.leader_active = False
                self.leader_status = f'{self.leader_read_error or "STALE"} - PRESS F11'
                self.controller.reset()
            active = self.enabled and start - self.last_input <= INPUT_TIMEOUT and self.fault is None
            if leader_mode:
                active = bool(self.leader_active and self.fault is None)
            if start - self.last_input > INPUT_TIMEOUT and self.enabled:
                self.enabled = False
                self.needs_release = True
                active = False
            if start - deadline > INPUT_TIMEOUT:
                self._reset_input()
                active = False
            recorded_before = None
            if self.recorder is not None and self.recording_error is None:
                try:
                    recorded_before = capture_state(self)
                except Exception as exc:
                    self.recording_error = f'{type(exc).__name__}: {exc}'
            r = None
            try:
                if self.fault is None:
                    s = self.backend.state()
                    if leader_mode:
                        if active:
                            r = self._leader_command(s, self.leader_unwrapped)
                        else:
                            self.backend.command(s['q'], self.gripper_target)
                        self.result = r
                    else:
                        if self.native_hold_on_release:
                            rotation = np.asarray(self.rotation_input) if active else np.zeros(3)
                            # Re-anchor on release, reversal or a change of rotation
                            # axis so a previous angular target cannot keep driving.
                            changed_rotation = (np.any(self._native_previous_rotation)
                                                and np.dot(self._native_previous_rotation, rotation) <= 0)
                            if not active and self.controller.was_enabled:
                                self.controller.reset()
                            elif changed_rotation:
                                self.controller.target_rotation = s['rotation'].copy()
                            self._native_previous_rotation = rotation.copy()
                        r = self.controller.step(s["q"], s["qvel"], s["tcp"], s["rotation"], s["jacobian"],
                                                 self.direction, self.rotation_input, bool(active), DT)
                        if active:
                            self.gripper_target = float(np.clip(self.gripper_target + self.gripper_input * .55 * DT, 0, 1))
                        self.backend.command(r["q_ref"], self.gripper_target)
                        self.result = r
                self.backend.step(self.physics_steps, return_state=False)
            except Exception as exc:
                self.fault = f"{type(exc).__name__}: {exc}"
                self.enabled = False
                self.needs_release = True
                self.leader_active = False
                self.leader_status = 'SIMULATION FAULT'
                # Stop issuing the old moving reference, then keep physics alive.
                s = self.backend.state()
                self.backend.command(s["q"], self.gripper_target)
            if recorded_before is not None and self.recording_error is None:
                try:
                    action = {'direction': list(self.direction) if active else [0.] * 3,
                              'rotation': list(self.rotation_input) if active else [0.] * 3,
                              'gripper': self.gripper_input if active else 0,
                              'enabled': bool(active),
                              'requested_direction': list(self.direction),
                              'requested_rotation': list(self.rotation_input),
                              'requested_gripper': self.gripper_input,
                              'input_stale': start - self.last_input > INPUT_TIMEOUT,
                              'q_ref': None if r is None else r['q_ref'].copy(),
                              'ctrl': self.backend.data.ctrl.copy(),
                              'gripper_target': self.gripper_target,
                              'constraints': [] if r is None else list(r['constraints']),
                              'fault': self.fault}
                    if leader_mode:
                        action.update(source='leader', direction=[0.] * 3, rotation=[0.] * 3,
                                      gripper=0, input_stale=not leader_fresh,
                                      leader_sample=None if leader_sample is None else
                                      {'positions': leader_sample[0].copy(), 'timestamp': leader_sample[1]})
                    self.recorder.append(recorded_before, action, capture_state(self))
                except Exception as exc:
                    # Data collection must not interrupt mechanical control.
                    self.recording_error = f'{type(exc).__name__}: {exc}'
            elapsed = (time.monotonic() - start) * 1000
            self.tick_ms.append(elapsed)
            self.tick_times.append(start)
            if elapsed > DT * 1000:
                self.overruns += 1

    def run(self):
        # Group physics steps into one 10 ms control tick.
        deadline = time.monotonic()
        while not self.stop_event.is_set():
            start = time.monotonic()
            self.tick(start, deadline)
            deadline += DT
            now = time.monotonic()
            if now > deadline:
                deadline = now
            time.sleep(max(0, deadline - time.monotonic()))


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, simulation):
        self.simulation = simulation
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server: Server

    def log_message(self, *_args):
        pass

    def send_body(self, status, data, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def json_response(self, status, data):
        self.send_body(status, json.dumps(data, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/state":
            self.json_response(200, self.server.simulation.snapshot())
            return
        files = {"/": ("index.html", "text/html"), "/index.html": ("index.html", "text/html"),
                 "/app.js": ("app.js", "text/javascript"), "/input-settings.js": ("input-settings.js", "text/javascript"),
                 "/styles.css": ("styles.css", "text/css")}
        if path not in files:
            self.json_response(404, {"error": "not found"})
            return
        name, mime = files[path]
        self.send_body(200, (STATIC / name).read_bytes(), mime + "; charset=utf-8")

    def do_POST(self):
        port = self.server.server_address[1]
        origin = self.headers.get("Origin")
        if origin is not None and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
            self.json_response(403, {"error": "cross-origin control is not allowed"})
            return
        if self.headers.get_content_type() != "application/json":
            self.json_response(415, {"error": "application/json is required"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("invalid request size")
            def reject_constant(value):
                raise ValueError(f"non-finite JSON value: {value}")
            body = json.loads(self.rfile.read(length), parse_constant=reject_constant)
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            path = urlsplit(self.path).path
            if path == "/api/input":
                self.server.simulation.set_input(body)
            elif path == "/api/config":
                self.server.simulation.configure(body)
            elif path == "/api/reset":
                if body:
                    raise ValueError("reset body must be empty")
                self.server.simulation.reset()
            else:
                self.json_response(404, {"error": "not found"})
                return
            self.json_response(200, {"ok": True})
        except (ValueError, TypeError, UnicodeDecodeError) as exc:
            self.json_response(400, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description="Gamepad arm teleoperation - local simulation only")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    simulation = Simulation()
    server = Server(("127.0.0.1", args.port), simulation)
    simulation.start()
    print(f"Arm Teleop Lab: http://127.0.0.1:{args.port}  (simulation only)", flush=True)
    print("Hold gamepad LB or keyboard Space to enable. Ctrl+C stops the server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        simulation.close()
        server.server_close()


if __name__ == "__main__":
    main()

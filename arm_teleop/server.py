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

from .mujoco_backend import MujocoBackend
from .pose_control import PoseController

DT = 0.01
INPUT_TIMEOUT = 0.25
STATIC = Path(__file__).parent / "static"


def input_vector(value):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("direction must contain three numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) or abs(x) > 1 for x in value):
        raise ValueError("direction values must be finite numbers in [-1, 1]")
    return [float(x) for x in value]


class Simulation:
    """One physics owner thread; commands and snapshots are protected by a lock."""
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.method = "dls"
        self.elbow_weight = .35
        self.backend = MujocoBackend("ur5e")
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

    def configure(self, body):
        if not body or not set(body) <= {"method", "elbow_weight", "robot"}:
            raise ValueError("expected method, robot and/or elbow_weight")
        if "method" in body and body["method"] not in ("dls", "sdls", "position_priority"):
            raise ValueError("unknown controller method")
        if "robot" in body and body["robot"] not in ("ur5e", "panda"):
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
                self.backend = new_backend
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
            result = dict(joints=s["q"].tolist(), joint_velocity=s["qvel"].tolist(),
                          points=s["points"].tolist(), tcp=s["tcp"].tolist(), target=target.tolist(),
                          tcp_rotation=s["rotation"].tolist(), tcp_quat=quat.tolist(),
                          target_rotation=(s["rotation"] if r is None else r["target_rotation"]).tolist(),
                          error_m=float(np.linalg.norm(target - s["tcp"])),
                          orientation_error_rad=0 if r is None else r["error_rad"],
                          solve_ms=0 if r is None else r["solve_ms"],
                          sigma_min=float(np.linalg.svd(s["jacobian"][:3], compute_uv=False)[-1]),
                          constraints=["hold"] if r is None else r["constraints"],
                          method=self.method, controller_mode=self.method, robot_id=self.backend.name,
                          elbow_weight=self.elbow_weight, gripper_position=s["gripper_position"],
                          gripper_range=s["gripper_range"], gripper_target=self.gripper_target,
                          finger_positions=s["finger_positions"].tolist(),
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

    def run(self):
        deadline = time.monotonic()
        substep = 0
        while not self.stop_event.is_set():
            start = time.monotonic()
            with self.lock:
                dt_physics = self.backend.timestep
                active = self.enabled and start - self.last_input <= INPUT_TIMEOUT and self.fault is None
                if start - self.last_input > INPUT_TIMEOUT and self.enabled:
                    self.enabled = False
                    self.needs_release = True
                    active = False
                if start - deadline > INPUT_TIMEOUT:
                    self._reset_input()
                    active = False
                try:
                    if substep == 0 and self.fault is None:
                        s = self.backend.state()
                        r = self.controller.step(s["q"], s["qvel"], s["tcp"], s["rotation"], s["jacobian"],
                                                 self.direction, self.rotation_input, bool(active), DT)
                        if active:
                            self.gripper_target = float(np.clip(self.gripper_target + self.gripper_input * .55 * DT, 0, 1))
                        self.backend.command(r["q_ref"], self.gripper_target)
                        self.result = r
                    self.backend.step()
                except Exception as exc:
                    self.fault = f"{type(exc).__name__}: {exc}"
                    self.enabled = False
                    self.needs_release = True
                    # Stop issuing the old moving reference, then keep physics alive.
                    s = self.backend.state()
                    self.backend.command(s["q"], self.gripper_target)
                elapsed = (time.monotonic() - start) * 1000
                if substep == 0:
                    self.tick_ms.append(elapsed)
                    self.tick_times.append(start)
                if elapsed > dt_physics * 1000:
                    self.overruns += 1
                substep = (substep + 1) % 5
            deadline += dt_physics
            now = time.monotonic()
            if now > deadline:
                deadline = now
            self.stop_event.wait(max(0, deadline - time.monotonic()))


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
                 "/app.js": ("app.js", "text/javascript"), "/styles.css": ("styles.css", "text/css")}
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

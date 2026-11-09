"""Streaming, local demonstration recordings for the simulation.

JSONL keeps a readable prefix if the application is interrupted. A missing end
record means the episode is incomplete. Disk writes run on a bounded queue so a
slow disk never blocks the physics loop indefinitely.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from queue import Full, Queue
import subprocess
import threading
import uuid

import mujoco
import numpy as np


class RecordingError(RuntimeError):
    """The recording is incomplete and should not be used for training."""


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot record {type(value).__name__}')


def _line(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'), default=_json_default) + '\n'


def _sha256(path):
    try:
        # Git checkouts may use CRLF on Windows and LF on Linux.
        return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    except OSError:
        return None


def model_fingerprints(robot_id):
    """Hash the installed source profile and XML, never the temporary scene."""
    root = Path(__file__).resolve().parent.parent
    profile = root / 'robot_models' / robot_id / 'profile.json'
    profile_hash = _sha256(profile)
    try:
        model = profile.parent / json.loads(profile.read_text(encoding='utf-8'))['model']
        model_hash = _sha256(model)
    except (OSError, ValueError, KeyError, TypeError):
        model_hash = None
    return {'profile_sha256': profile_hash, 'model_xml_sha256': model_hash}


def recording_metadata(simulation, scene_id, control_dt):
    """Metadata for the current model; call while it is not being replaced."""
    b, c = simulation.backend, simulation.controller
    tuning = ('position_gain', 'orientation_gain', 'cartesian_speed', 'angular_speed',
              'length_scale', 'damping', 'max_position_lead', 'max_orientation_lead',
              'max_joint_reference_lead', 'posture_gain', 'sdls_gamma')
    root = Path(__file__).resolve().parent.parent
    code_files = ('arm_teleop/server.py', 'arm_teleop/b601_scene.py', 'arm_teleop/leader_serial.py',
                  'optimized_algorithm/pose_control.py', 'optimized_algorithm/native_control.py',
                  'scripts/native_viewer.py', 'scripts/native_input.py')
    try:
        revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, timeout=2,
                                  check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {'robot_id': b.name, 'scene_id': scene_id,
            'mujoco_version': mujoco.__version__, 'control_dt': float(control_dt),
            'physics_timestep': float(b.timestep),
            'model_fingerprints': model_fingerprints(b.name),
            'git_revision': revision,
            'code_sha256': {name: _sha256(root / name) for name in code_files},
            'model_dimensions': {'nq': b.model.nq, 'nv': b.model.nv,
                                 'na': b.model.na, 'nu': b.model.nu},
            'joint_names': list(b.profile['joints']),
            'actuator_names': list(b.profile['actuators']),
            'controller': {'mode': c.mode, **{k: getattr(c, k) for k in tuning}},
            'units': {'position': 'm', 'joint_angle': 'rad', 'time': 's',
                      'quaternion': 'wxyz', 'frame': 'MuJoCo world'}}


def _copy(value):
    return None if value is None else np.asarray(value).copy()


def capture_state(simulation):
    """Copy the current state while the caller holds the simulation lock.

    No Jacobian or rendering is computed. MuJoCo arrays and controller state
    are copied so the background writer cannot observe a later physics tick.
    """
    b, c = simulation.backend, simulation.controller
    m, d = b.model, b.data
    cube_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'practice_cube_free')
    cube = None
    if cube_id >= 0:
        p, v = int(m.jnt_qposadr[cube_id]), int(m.jnt_dofadr[cube_id])
        cube = {'position': d.qpos[p:p+3].copy(), 'quaternion': d.qpos[p+3:p+7].copy(),
                'linear_velocity': d.qvel[v:v+3].copy(),
                'angular_velocity_local': d.qvel[v+3:v+6].copy()}
    fingers = d.qpos[b.finger_qpos_ids].copy()
    opening = (float(np.clip(np.mean((fingers - b.gripper_closed) /
                                      (b.gripper_open - b.gripper_closed)), 0, 1))
               if len(fingers) else 0.)
    return {'mj': {'time': float(d.time), 'qpos': d.qpos.copy(), 'qvel': d.qvel.copy(),
                   'act': d.act.copy(), 'ctrl': d.ctrl.copy(),
                   'mocap_pos': d.mocap_pos.copy(), 'mocap_quat': d.mocap_quat.copy()},
            'robot': {'joint_position': d.qpos[b.qpos_ids].copy(),
                      'joint_velocity': d.qvel[b.dof_ids].copy(),
                      'tcp_position': d.site_xpos[b.tcp_id].copy(),
                      'tcp_rotation': d.site_xmat[b.tcp_id].reshape(3, 3).copy(),
                      'finger_position': fingers, 'gripper_opening': opening,
                      'actuator_force': d.actuator_force.copy()},
            'cube': cube,
            'controller': {'target_position': _copy(c.target_position),
                           'target_rotation': _copy(c.target_rotation),
                           'q_ref': _copy(c.q_ref), 'velocity': _copy(c.velocity),
                           'was_enabled': bool(c.was_enabled),
                           'gripper_target': float(simulation.gripper_target),
                           'previous_rotation': _copy(simulation._native_previous_rotation),
                           'fault': simulation.fault}}


class EpisodeRecorder:
    """One episode; a completed file always ends with an explicit outcome."""

    def __init__(self, directory, metadata, queue_size=1000):
        if queue_size < 1:
            raise ValueError('queue_size must be positive')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        self.path = directory / f"episode_{now.strftime('%Y%m%dT%H%M%S_%fZ')}_{uuid.uuid4().hex[:8]}.jsonl"
        self._file = self.path.open('x', encoding='utf-8', newline='\n')
        try:
            self._file.write(_line({**metadata, 'type': 'metadata', 'schema_version': 1,
                                    'created_utc': now.isoformat()}))
            self._file.flush()
        except Exception:
            self._file.close()
            raise
        self._queue = Queue(maxsize=queue_size)
        self._error = None
        self._closed = False
        self.steps = 0
        self._thread = threading.Thread(target=self._write, name='episode-writer', daemon=True)
        self._thread.start()

    @property
    def error(self):
        return self._error

    def _write(self):
        try:
            while True:
                row = self._queue.get()
                if row is None:
                    break
                self._file.write(_line(row))
                # Buffered writes bound data loss if the app is interrupted.
                if row['type'] == 'end' or row.get('index', -1) % 100 == 99:
                    self._file.flush()
        except (OSError, TypeError, ValueError) as exc:
            self._error = f'{type(exc).__name__}: {exc}'
        finally:
            try:
                self._file.close()
            except OSError as exc:
                self._error = f'{type(exc).__name__}: {exc}'

    def append(self, state, action, next_state):
        if self._closed:
            raise RecordingError('episode is already closed')
        if self._error:
            raise RecordingError(self._error)
        row = {'type': 'transition', 'index': self.steps,
               'state': state, 'action': action, 'next_state': next_state}
        try:
            self._queue.put_nowait(row)
        except Full as exc:
            self._error = 'recording queue is full; episode is incomplete'
            raise RecordingError(self._error) from exc
        self.steps += 1

    def finish(self, outcome, reason):
        """Finish explicitly; interrupted/overflowed episodes remain incomplete."""
        if self._closed:
            return
        if outcome not in ('success', 'failure', 'interrupted'):
            raise ValueError('outcome must be success, failure or interrupted')
        if not isinstance(reason, str) or not reason:
            raise ValueError('reason must be a nonempty string')
        self._closed = True
        # A failed writer may stop with a full queue: do not wait on it forever.
        def enqueue(row):
            while self._thread.is_alive():
                try:
                    self._queue.put(row, timeout=.1)
                    return
                except Full:
                    continue
        if not self._error:
            enqueue({'type': 'end', 'outcome': outcome, 'reason': reason,
                     'steps': self.steps})
        enqueue(None)
        self._thread.join()
        if self._error:
            raise RecordingError(self._error)

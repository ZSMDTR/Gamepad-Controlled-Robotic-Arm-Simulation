"""Configuration-driven MuJoCo adapter for fixed-base arm simulations.

Profiles map an MJCF model's joints, position actuators and TCP to the
controller. They describe simulation only, not a real-robot safety controller.
"""
import json
from pathlib import Path

import mujoco
import numpy as np

_MODEL_ROOT = Path(__file__).resolve().parent.parent / 'robot_models'


def available_models():
    """Return model directory names with a profile, for UI selection."""
    return sorted(p.name for p in _MODEL_ROOT.iterdir()
                  if p.is_dir() and (p / 'profile.json').is_file())


def _names(value, name, allow_empty=False):
    if (not isinstance(value, list) or (not value and not allow_empty)
            or any(not isinstance(x, str) or not x for x in value)
            or len(set(value)) != len(value)):
        raise ValueError(f'{name} must contain unique, nonempty names')
    return value


def _vector(value, size, name, positive=False):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite vector of length {size}') from exc
    if result.shape != (size,) or not np.isfinite(result).all() or (positive and np.any(result <= 0)):
        raise ValueError(f'{name} must be a finite {"positive " if positive else ""}vector of length {size}')
    return result


class MujocoBackend:
    def __init__(self, model_name='ur5e', profile_path=None):
        # model_name remains the UI identifier; profile_path enables testing or
        # integrating a profile outside the bundled model directories.
        if profile_path is None:
            if model_name not in available_models():
                raise ValueError(f'Unknown model {model_name!r}; available: {available_models()}')
            profile_path = _MODEL_ROOT / model_name / 'profile.json'
        profile_path = Path(profile_path).resolve()
        try:
            with profile_path.open(encoding='utf-8') as f:
                p = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'Cannot read robot profile {profile_path}: {exc}') from exc
        if not isinstance(p, dict) or not isinstance(p.get('model'), str) or not p['model']:
            raise ValueError('profile requires a model path')
        self.name = model_name
        self.profile = p
        model_path = (profile_path.parent / p['model']).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        m = self.model

        def ids(kind, names):
            result = [mujoco.mj_name2id(m, kind, name) for name in names]
            missing = [name for name, i in zip(names, result) if i < 0]
            if missing:
                raise ValueError(f'Model {model_name} is missing named elements: {missing}')
            return np.array(result, dtype=int)

        joints = _names(p.get('joints'), 'joints')
        actuators = _names(p.get('actuators'), 'actuators')
        n = len(joints)
        if len(actuators) != n:
            raise ValueError('actuators must match joints one-to-one')
        velocity = _vector(p.get('velocity'), n, 'velocity', positive=True)
        acceleration = _vector(p.get('acceleration'), n, 'acceleration', positive=True)
        self.joint_ids = ids(mujoco.mjtObj.mjOBJ_JOINT, joints)
        self.qpos_ids = m.jnt_qposadr[self.joint_ids].copy()
        self.dof_ids = m.jnt_dofadr[self.joint_ids].copy()
        self.actuator_ids = ids(mujoco.mjtObj.mjOBJ_ACTUATOR, actuators)
        if any(m.jnt_type[i] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
               or not m.jnt_limited[i] for i in self.joint_ids):
            raise ValueError('arm joints must be limited scalar hinge or slide joints')
        for joint, actuator in zip(self.joint_ids, self.actuator_ids):
            # q_ref is in joint coordinates, so a direct unit-gear position
            # servo is required; torque and tendon controls are incompatible.
            if (m.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT
                    or m.actuator_trnid[actuator, 0] != joint
                    or not np.allclose(m.actuator_gear[actuator], [1, 0, 0, 0, 0, 0])
                    or m.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED
                    or m.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE
                    or m.actuator_gainprm[actuator, 0] <= 0
                    or not np.isclose(m.actuator_biasprm[actuator, 0], 0)
                    or not np.isclose(m.actuator_biasprm[actuator, 1], -m.actuator_gainprm[actuator, 0])
                    or not m.actuator_ctrllimited[actuator]):
                raise ValueError('each arm actuator must directly position-control its matching joint with unit gear and control limits')
        tcp = p.get('tcp', 'tcp')
        if not isinstance(tcp, str) or not tcp:
            raise ValueError('tcp must be a site name')
        self.tcp_id = int(ids(mujoco.mjtObj.mjOBJ_SITE, [tcp])[0])
        # Every movable joint from world to TCP must be explicitly controlled,
        # in base-to-tool order. This also rejects a floating-base robot.
        chain = []
        body = int(m.site_bodyid[self.tcp_id])
        while body:
            start = int(m.body_jntadr[body])
            chain[0:0] = list(range(start, start + int(m.body_jntnum[body])))
            body = int(m.body_parentid[body])
        if chain != self.joint_ids.tolist():
            raise ValueError('joints must list every joint on the fixed-base TCP chain in base-to-tool order')
        lower = np.maximum(m.jnt_range[self.joint_ids, 0], m.actuator_ctrlrange[self.actuator_ids, 0])
        upper = np.minimum(m.jnt_range[self.joint_ids, 1], m.actuator_ctrlrange[self.actuator_ids, 1])
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower >= upper):
            raise ValueError('joint and actuator limits must have a finite nonempty intersection')
        self.spec = {'lower': lower, 'upper': upper,
                     'velocity_limit': velocity, 'acceleration_limit': acceleration}

        g = p.get('gripper')
        if g is None:
            g = {'joints': [], 'bodies': [], 'closed': [], 'open': [],
                 'actuators': [], 'control_closed': [], 'control_open': []}
        if not isinstance(g, dict):
            raise ValueError('gripper must be an object or omitted')
        finger_names = _names(g.get('joints'), 'gripper joints', allow_empty=True)
        body_names = _names(g.get('bodies'), 'gripper bodies', allow_empty=True)
        gripper_names = _names(g.get('actuators'), 'gripper actuators', allow_empty=True)
        if len(body_names) != len(finger_names) or bool(finger_names) != bool(gripper_names):
            raise ValueError('gripper needs matching joints and bodies, plus actuators')
        self.finger_ids = ids(mujoco.mjtObj.mjOBJ_JOINT, finger_names)
        if any(m.jnt_type[i] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
               for i in self.finger_ids):
            raise ValueError('gripper joints must be scalar hinge or slide joints')
        self.finger_qpos_ids = m.jnt_qposadr[self.finger_ids].copy()
        self.finger_body_ids = ids(mujoco.mjtObj.mjOBJ_BODY, body_names)
        self.gripper_actuator_ids = ids(mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_names)
        self.gripper_closed = _vector(g.get('closed'), len(finger_names), 'gripper closed')
        self.gripper_open = _vector(g.get('open'), len(finger_names), 'gripper open')
        if np.any(self.gripper_open == self.gripper_closed):
            raise ValueError('gripper open and closed positions must differ')
        self.control_closed = _vector(g.get('control_closed'), len(gripper_names), 'gripper control_closed')
        self.control_open = _vector(g.get('control_open'), len(gripper_names), 'gripper control_open')
        for idx, closed, opened in zip(self.gripper_actuator_ids, self.control_closed, self.control_open):
            if not m.actuator_ctrllimited[idx] or min(closed, opened) < m.actuator_ctrlrange[idx, 0] or max(closed, opened) > m.actuator_ctrlrange[idx, 1]:
                raise ValueError('gripper controls must fit the actuator control range')
        self.home_key = p.get('home_key', 'home')
        if not isinstance(self.home_key, str) or not self.home_key:
            raise ValueError('home_key must be a keyframe name')
        self.reset()
        self.spec['home'] = self.data.qpos[self.qpos_ids].copy()
        if np.any(self.spec['home'] < lower) or np.any(self.spec['home'] > upper):
            raise ValueError('home keyframe must be inside the arm limits')

    @property
    def timestep(self):
        return float(self.model.opt.timestep)

    def reset(self):
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, self.home_key)
        if key < 0:
            raise ValueError(f'Robot model is missing keyframe {self.home_key!r}')
        mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def command(self, q_ref, gripper_fraction):
        q_ref = np.asarray(q_ref, dtype=float)
        if q_ref.shape != self.spec['lower'].shape or not np.isfinite(q_ref).all():
            raise ValueError('q_ref must be a finite value for each arm joint')
        if not np.isfinite(gripper_fraction):
            raise ValueError('gripper_fraction must be finite')
        self.data.ctrl[self.actuator_ids] = np.clip(q_ref, self.spec['lower'], self.spec['upper'])
        fraction = np.clip(gripper_fraction, 0., 1.)
        self.data.ctrl[self.gripper_actuator_ids] = self.control_closed + fraction * (self.control_open - self.control_closed)

    def step(self, steps=1, return_state=True):
        if not isinstance(steps, int) or steps < 1:
            raise ValueError('steps must be a positive integer')
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        # Refresh kinematics after integration before another state() call.
        mujoco.mj_forward(self.model, self.data)
        return self.state() if return_state else None

    def state(self):
        m, d = self.model, self.data
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, self.tcp_id)
        tcp = d.site_xpos[self.tcp_id].copy()
        rotation = d.site_xmat[self.tcp_id].reshape(3, 3).copy()
        fingers = d.qpos[self.finger_qpos_ids].copy()
        opening = float(np.clip(np.mean((fingers - self.gripper_closed) / (self.gripper_open - self.gripper_closed)), 0., 1.)) if len(fingers) else 0.
        base = d.xpos[m.body_parentid[m.jnt_bodyid[self.joint_ids[0]]]]
        points = np.vstack((base, d.xanchor[self.joint_ids], tcp))
        return {'q': d.qpos[self.qpos_ids].copy(), 'qvel': d.qvel[self.dof_ids].copy(),
                'tcp': tcp, 'rotation': rotation,
                'jacobian': np.vstack((jp[:, self.dof_ids], jr[:, self.dof_ids])),
                'points': points,
                'gripper_position': opening, 'gripper_range': [0., 1.],
                'finger_positions': d.xpos[self.finger_body_ids].copy(),
                'finger_q': fingers, 'time': float(d.time),
                'actuator_force': d.actuator_force[self.actuator_ids].copy(),
                'joint_lower': self.spec['lower'].copy(), 'joint_upper': self.spec['upper'].copy()}

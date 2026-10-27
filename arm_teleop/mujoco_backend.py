"""MuJoCo dynamics adapter for fixed-base UR5e and Panda simulations.

The position actuators are set through ctrl and mj_step owns the state. Model
assets are derived from MuJoCo Menagerie; see robot_models/*/SOURCE.txt.
This adapter is for simulation, not a real-robot safety controller.
"""
from pathlib import Path

import mujoco
import numpy as np

_MODELS = {
    'ur5e': {
        'joints': ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                   'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
        'actuators': ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3'],
        'fingers': ['demo_left_joint', 'demo_right_joint'],
        'finger_bodies': ['demo_left', 'demo_right'],
        'gripper_actuators': ['demo_left_actuator', 'demo_right_actuator'],
        'gripper_max': .03, 'gripper_ctrl_max': .03,
        # Conservative simulation profile, not a manufacturer's specifications.
        'velocity': [.8, .8, .8, 1., 1., 1.],
        'acceleration': [2., 2., 2., 3., 3., 3.],
    },
    'panda': {
        'joints': [f'joint{i}' for i in range(1, 8)],
        'actuators': [f'actuator{i}' for i in range(1, 8)],
        'fingers': ['finger_joint1', 'finger_joint2'],
        'finger_bodies': ['left_finger', 'right_finger'],
        'gripper_actuators': ['actuator8'],
        'gripper_max': .04, 'gripper_ctrl_max': 255.,
        'velocity': [.8, .8, .8, .8, 1., 1., 1.],
        'acceleration': [2., 2., 2., 2., 3., 3., 3.],
    },
}

class MujocoBackend:
    def __init__(self, model_name='ur5e'):
        if model_name not in _MODELS:
            raise ValueError(f'Unknown model {model_name!r}; choose ur5e or panda')
        self.name = model_name
        self.profile = _MODELS[model_name]
        model_path = Path(__file__).resolve().parent.parent / 'robot_models' / model_name / 'model.xml'
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        m, p = self.model, self.profile
        def ids(kind, names):
            result = [mujoco.mj_name2id(m, kind, name) for name in names]
            if any(i < 0 for i in result):
                raise ValueError(f'Model {model_name} is missing named elements: {names}')
            return np.array(result, dtype=int)
        self.joint_ids = ids(mujoco.mjtObj.mjOBJ_JOINT, p['joints'])
        self.qpos_ids = m.jnt_qposadr[self.joint_ids].copy()
        self.dof_ids = m.jnt_dofadr[self.joint_ids].copy()
        self.actuator_ids = ids(mujoco.mjtObj.mjOBJ_ACTUATOR, p['actuators'])
        self.finger_ids = ids(mujoco.mjtObj.mjOBJ_JOINT, p['fingers'])
        self.finger_qpos_ids = m.jnt_qposadr[self.finger_ids].copy()
        self.finger_body_ids = ids(mujoco.mjtObj.mjOBJ_BODY, p['finger_bodies'])
        self.gripper_actuator_ids = ids(mujoco.mjtObj.mjOBJ_ACTUATOR, p['gripper_actuators'])
        self.tcp_id = int(ids(mujoco.mjtObj.mjOBJ_SITE, ['tcp'])[0])
        lower = np.maximum(m.jnt_range[self.joint_ids, 0], m.actuator_ctrlrange[self.actuator_ids, 0])
        upper = np.minimum(m.jnt_range[self.joint_ids, 1], m.actuator_ctrlrange[self.actuator_ids, 1])
        self.spec = {'lower': lower, 'upper': upper,
                     'velocity_limit': np.array(p['velocity']),
                     'acceleration_limit': np.array(p['acceleration'])}
        self.reset()
        self.spec['home'] = self.data.qpos[self.qpos_ids].copy()

    @property
    def timestep(self):
        return float(self.model.opt.timestep)

    def reset(self):
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if key < 0:
            raise ValueError('Robot model is missing its home keyframe')
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
        self.data.ctrl[self.gripper_actuator_ids] = np.clip(gripper_fraction, 0., 1.) * self.profile['gripper_ctrl_max']

    def step(self, steps=1):
        if not isinstance(steps, int) or steps < 1:
            raise ValueError('steps must be a positive integer')
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        # mj_step updates qpos at the end of integration; refresh kinematics for
        # the newly integrated state before reading TCP and Jacobians.
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        m, d = self.model, self.data
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, self.tcp_id)
        tcp = d.site_xpos[self.tcp_id].copy()
        rotation = d.site_xmat[self.tcp_id].reshape(3, 3).copy()
        fingers = d.qpos[self.finger_qpos_ids].copy()
        opening = float(np.clip(fingers.mean() / self.profile['gripper_max'], 0., 1.))
        points = np.vstack((np.zeros(3), d.xanchor[self.joint_ids], tcp))
        return {'q': d.qpos[self.qpos_ids].copy(), 'qvel': d.qvel[self.dof_ids].copy(),
                'tcp': tcp, 'rotation': rotation,
                'jacobian': np.vstack((jp[:, self.dof_ids], jr[:, self.dof_ids])),
                'points': points,
                'gripper_position': opening, 'gripper_range': [0., 1.],
                'finger_positions': d.xpos[self.finger_body_ids].copy(),
                'finger_q': fingers, 'time': float(d.time),
                'actuator_force': d.actuator_force[self.actuator_ids].copy(),
                'joint_lower': self.spec['lower'].copy(), 'joint_upper': self.spec['upper'].copy()}

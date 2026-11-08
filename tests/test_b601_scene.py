import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from arm_teleop.b601_scene import make_b601_scene_profile
from arm_teleop.mujoco_backend import MujocoBackend
from arm_teleop.pose_control import PoseController, rotation_log


class B601PracticeSceneTest(unittest.TestCase):
    def test_scene_keeps_robot_and_cube_stable(self):
        profile = Path(__file__).resolve().parents[1] / 'robot_models/b601_rs/profile.json'
        if not profile.is_file():
            self.skipTest('B601 model not installed')
        with tempfile.TemporaryDirectory() as folder:
            scene_profile = make_b601_scene_profile(profile, folder)
            backend = MujocoBackend('b601_rs', scene_profile)
            model, data = backend.model, backend.data
            cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'practice_cube')
            table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'practice_table')
            self.assertGreater(cube, 0)
            self.assertGreaterEqual(table, 0)
            backend.step(500, return_state=False)
            self.assertTrue(np.isfinite(data.qpos).all())
            self.assertAlmostEqual(data.xpos[cube, 2], .282, delta=.003)
            self.assertAlmostEqual(data.xpos[cube, 0], .4, delta=.003)

    def test_fingers_can_lift_and_hold_cube(self):
        profile = Path(__file__).resolve().parents[1] / 'robot_models/b601_rs/profile.json'
        if not profile.is_file():
            self.skipTest('B601 model not installed')
        with tempfile.TemporaryDirectory() as folder:
            backend = MujocoBackend('b601_rs', make_b601_scene_profile(profile, folder))
            model, data = backend.model, backend.data
            # Start at a manually determined pre-grasp pose; this tests physics,
            # not navigation to the pose or actual hand-held controller input.
            q = np.array([0, 1.17, .83, .09, 0, 0])
            data.qpos[backend.qpos_ids] = q
            data.ctrl[backend.actuator_ids] = q
            coupler = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint7')
            data.qpos[model.jnt_qposadr[coupler]] = .05
            data.qpos[backend.finger_qpos_ids] = .05
            gripper = backend.gripper_actuator_ids[0]
            data.ctrl[gripper] = .05
            mujoco.mj_forward(model, data)
            for _ in range(200):
                mujoco.mj_step(model, data)
            data.ctrl[gripper] = 0
            for _ in range(1500):
                mujoco.mj_step(model, data)
            cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'practice_cube_geom')
            contacts = set()
            for contact in data.contact[:data.ncon]:
                if cube_geom in (contact.geom1, contact.geom2):
                    other = contact.geom2 if contact.geom1 == cube_geom else contact.geom1
                    contacts.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or '')
            self.assertTrue(any(name.startswith('gripper_left_') for name in contacts))
            self.assertTrue(any(name.startswith('gripper_right_') for name in contacts))
            for i in range(1000):
                data.ctrl[backend.actuator_ids[1]] = q[1] - .00015 * i
                mujoco.mj_step(model, data)
            for _ in range(1000):
                mujoco.mj_step(model, data)
            cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'practice_cube')
            self.assertGreater(data.xpos[cube, 2], .325)
            data.ctrl[gripper] = .05
            for _ in range(1500):
                mujoco.mj_step(model, data)
            self.assertAlmostEqual(data.xpos[cube, 2], .282, delta=.003)

    def test_controller_reaches_and_grasps_from_home(self):
        profile = Path(__file__).resolve().parents[1] / 'robot_models/b601_rs/profile.json'
        if not profile.is_file():
            self.skipTest('B601 model not installed')
        with tempfile.TemporaryDirectory() as folder:
            backend = MujocoBackend('b601_rs', make_b601_scene_profile(profile, folder))
            model, data = backend.model, backend.data
            # Determine a known reachable grasp pose without mutating the live state.
            goal_data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, goal_data, 0)
            goal_data.qpos[backend.qpos_ids] = [0, 1.17, .83, .09, 0, 0]
            mujoco.mj_forward(model, goal_data)
            grasp_pos = goal_data.site_xpos[backend.tcp_id].copy()
            grasp_rot = goal_data.site_xmat[backend.tcp_id].reshape(3, 3).copy()
            controller = PoseController(dict(backend.spec, mode='dls', posture_gain=.07))
            home = backend.state()['tcp']
            cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'practice_cube')
            cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'practice_cube_geom')
            grip = 1.
            phases = [('open', home, 2), ('rotate', home, 10),
                      ('approach', grasp_pos, 20), ('close', grasp_pos, 3),
                      ('lift', grasp_pos + [0, 0, .06], 8),
                      ('carry', grasp_pos + [.05, 0, .06], 8),
                      ('release', grasp_pos + [.05, 0, .06], 3)]
            for phase, target, seconds in phases:
                for _ in range(seconds * 100):
                    state = backend.state()
                    error = target - state['tcp']
                    rot_error = rotation_log(grasp_rot @ state['rotation'].T)
                    translation = np.zeros(3) if phase == 'rotate' else np.clip(error * 3, -.8, .8)
                    rotation = np.clip(rot_error * 1.2, -.8, .8) if phase == 'rotate' else np.zeros(3)
                    if phase == 'close':
                        grip = max(0., grip - .55 * .01)
                    elif phase == 'release':
                        grip = min(1., grip + .55 * .01)
                    result = controller.step(state['q'], state['qvel'], state['tcp'], state['rotation'],
                                             state['jacobian'], translation, rotation, True, .01)
                    backend.command(result['q_ref'], grip)
                    backend.step(5, return_state=False)
                if phase == 'close':
                    contacts = set()
                    for contact in data.contact[:data.ncon]:
                        if cube_geom in (contact.geom1, contact.geom2):
                            other = contact.geom2 if contact.geom1 == cube_geom else contact.geom1
                            contacts.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or '')
                    self.assertTrue(any(name.startswith('gripper_left_') for name in contacts))
                    self.assertTrue(any(name.startswith('gripper_right_') for name in contacts))
                elif phase == 'lift':
                    self.assertGreater(data.xpos[cube, 2], .325)
                elif phase == 'carry':
                    self.assertGreater(data.xpos[cube, 0], .435)
                    self.assertGreater(data.xpos[cube, 2], .325)
                elif phase == 'release':
                    self.assertAlmostEqual(data.xpos[cube, 2], .282, delta=.003)


if __name__ == '__main__':
    unittest.main()

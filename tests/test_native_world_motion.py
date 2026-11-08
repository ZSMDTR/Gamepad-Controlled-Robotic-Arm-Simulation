"""Check world-axis motions against the B601 physics, not just input mapping."""
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from arm_teleop.b601_scene import make_b601_scene_profile
from arm_teleop.mujoco_backend import MujocoBackend
from arm_teleop.server import Simulation, physics_steps
from optimized_algorithm.native_control import make_native_controller
from optimized_algorithm.pose_control import rotation_log


class NativeWorldMotionTest(unittest.TestCase):
    def test_long_rotations_settle_when_stick_centers(self):
        profile = Path(__file__).resolve().parents[1] / 'robot_models/b601_rs/profile.json'
        if not profile.is_file():
            self.skipTest('B601 model not installed')
        with tempfile.TemporaryDirectory() as folder:
            scene_profile = make_b601_scene_profile(profile, folder)
            for axis in range(3):
                for sign in (-1, 1):
                    with self.subTest(axis=axis, sign=sign):
                        sim = Simulation()
                        sim.backend = MujocoBackend('b601_rs', scene_profile)
                        sim.physics_steps = physics_steps(sim.backend.timestep)
                        sim.controller = make_native_controller(sim.backend.spec)
                        sim.native_hold_on_release = True
                        initial = sim.backend.state()
                        direction = (np.eye(3)[axis] * sign).tolist()
                        max_drift = 0.
                        for tick in range(500):
                            command = direction if tick < 400 else [0., 0., 0.]
                            sim.set_input({'direction': [0., 0., 0.], 'rotation': command, 'enabled': True})
                            now = time.monotonic()
                            sim.tick(now, now)
                            state = sim.backend.state()
                            max_drift = max(max_drift, np.linalg.norm(state['tcp'] - initial['tcp']))
                            if tick == 399:
                                released = state['rotation']
                        self.assertIsNone(sim.fault)
                        self.assertLess(max_drift, .015)
                        self.assertLess(np.linalg.norm(rotation_log(state['rotation'] @ released.T)), .02)

    def test_rotation_neutral_and_reversal_keep_joint_reference_continuous(self):
        sim = Simulation()
        sim.controller = make_native_controller(sim.backend.spec)
        sim.native_hold_on_release = True
        for command in ([1., 0., 0.], [0., 0., 0.], [-1., 0., 0.], [0., 1., 0.]):
            previous = sim.result['q_ref'].copy() if sim.result is not None else None
            sim.set_input({'direction': [0., 0., 0.], 'rotation': command, 'enabled': True})
            now = time.monotonic()
            sim.tick(now, now)
            self.assertIsNone(sim.fault)
            if previous is not None:
                self.assertTrue(np.all(np.abs(sim.result['q_ref'] - previous) <= sim.controller.vmax * .01 + 1e-9))
            for _ in range(40):
                sim.set_input({'direction': [0., 0., 0.], 'rotation': command, 'enabled': True})
                now = time.monotonic()
                sim.tick(now, now)

    def test_rotation_center_and_neutral_hold_after_long_input(self):
        profile = Path(__file__).resolve().parents[1] / 'robot_models/b601_rs/profile.json'
        if not profile.is_file():
            self.skipTest('B601 model not installed')
        with tempfile.TemporaryDirectory() as folder:
            sim = Simulation()
            sim.backend = MujocoBackend('b601_rs', make_b601_scene_profile(profile, folder))
            sim.physics_steps = physics_steps(sim.backend.timestep)
            sim.controller = make_native_controller(sim.backend.spec)
            sim.native_hold_on_release = True
            initial = sim.backend.state()
            max_drift = 0.
            for _ in range(500):
                sim.set_input({'direction': [0., 0., 0.], 'rotation': [0., 1., 0.], 'enabled': True})
                now = time.monotonic()
                sim.tick(now, now)
                max_drift = max(max_drift, np.linalg.norm(sim.backend.state()['tcp'] - initial['tcp']))
            released = sim.backend.state()
            for _ in range(200):
                sim.set_input({'direction': [0., 0., 0.], 'rotation': [0., 0., 0.], 'enabled': True})
                now = time.monotonic()
                sim.tick(now, now)
                max_drift = max(max_drift, np.linalg.norm(sim.backend.state()['tcp'] - initial['tcp']))
            held = sim.backend.state()
            self.assertIsNone(sim.fault)
            self.assertLess(max_drift, .015)
            self.assertLess(np.linalg.norm(rotation_log(held['rotation'] @ released['rotation'].T)), .02)
            for _ in range(100):
                sim.set_input({'direction': [0., 0., 0.], 'rotation': [0., -1., 0.], 'enabled': True})
                now = time.monotonic()
                sim.tick(now, now)
            reversed_state = sim.backend.state()
            self.assertLess(rotation_log(reversed_state['rotation'] @ held['rotation'].T)[1], -.02)

    def test_six_translation_and_rotation_directions_from_home(self):
        profile = Path(__file__).resolve().parents[1] / 'robot_models/b601_rs/profile.json'
        if not profile.is_file():
            self.skipTest('B601 model not installed')
        with tempfile.TemporaryDirectory() as folder:
            scene_profile = make_b601_scene_profile(profile, folder)
            for rotation_mode in (False, True):
                for axis in range(3):
                    for sign in (-1, 1):
                        with self.subTest(rotation=rotation_mode, axis=axis, sign=sign):
                            backend = MujocoBackend('b601_rs', scene_profile)
                            controller = make_native_controller(backend.spec)
                            start = backend.state()
                            initial_pos, initial_rot = start['tcp'], start['rotation']
                            direction = np.eye(3)[axis] * sign
                            largest_cross_track = 0.
                            for tick in range(200):
                                state = backend.state()
                                command = direction if tick < 100 else np.zeros(3)
                                translation = np.zeros(3) if rotation_mode else command
                                rotation = command if rotation_mode else np.zeros(3)
                                result = controller.step(state['q'], state['qvel'], state['tcp'],
                                                         state['rotation'], state['jacobian'],
                                                         translation, rotation, True, .01)
                                backend.command(result['q_ref'], 1.)
                                backend.step(5, return_state=False)
                                displacement = backend.state()['tcp'] - initial_pos
                                deviation = np.linalg.norm(displacement if rotation_mode
                                                           else np.delete(displacement, axis))
                                largest_cross_track = max(largest_cross_track, deviation)
                            final = backend.state()
                            displacement = final['tcp'] - initial_pos
                            angle = rotation_log(final['rotation'] @ initial_rot.T)
                            if rotation_mode:
                                self.assertGreater(angle[axis] * sign, .2)
                                self.assertLess(np.linalg.norm(np.delete(angle, axis)), .04)
                                self.assertLess(largest_cross_track, .006)
                            else:
                                self.assertGreater(displacement[axis] * sign, .04)
                                self.assertLess(largest_cross_track, .006)
                                self.assertLess(np.linalg.norm(angle), .04)


if __name__ == '__main__':
    unittest.main()

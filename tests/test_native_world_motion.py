"""Check world-axis motions against the B601 physics, not just input mapping."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from arm_teleop.b601_scene import make_b601_scene_profile
from arm_teleop.mujoco_backend import MujocoBackend
from optimized_algorithm.native_control import make_native_controller
from optimized_algorithm.pose_control import rotation_log


class NativeWorldMotionTest(unittest.TestCase):
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

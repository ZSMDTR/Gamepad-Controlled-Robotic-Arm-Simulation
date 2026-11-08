import unittest

from scripts.native_input import InputMapper


class NativeInputTest(unittest.TestCase):
    def setUp(self):
        self.mapper = InputMapper()

    def test_keyboard_requires_release_after_start_and_focus_loss(self):
        sample = self.mapper.sample
        self.assertFalse(sample(True, {'SHIFT', 'W'}, None)['enabled'])
        self.assertFalse(sample(True, set(), None)['enabled'])
        self.assertEqual(sample(True, {'SHIFT', 'W'}, None)['direction'], [0, 1, 0])
        self.assertFalse(sample(False, {'SHIFT', 'W'}, None)['enabled'])
        self.assertFalse(sample(True, {'SHIFT', 'W'}, None)['enabled'])
        sample(True, set(), None)
        self.assertTrue(sample(True, {'SHIFT', 'W'}, None)['enabled'])
        self.assertFalse(sample(True, {'W'}, None)['enabled'])

    def test_gamepad_axis_recenter_mode_and_disconnect(self):
        sample = self.mapper.sample
        def pad(axes, *pressed):
            buttons = [int(i in pressed) for i in range(15)]
            return (0, axes, buttons)
        centered = [0., 0., 0., 0.]
        sample(True, set(), pad(centered))
        self.assertGreater(sample(True, set(), pad([.5, 0, 0, 0], 4))['direction'][0], 0)
        self.assertEqual(sample(True, set(), pad([.5, 0, 0, 0], 4, 5))['rotation'], [0, 0, 0])
        sample(True, set(), pad(centered, 4, 5))
        self.assertGreater(sample(True, set(), pad([.5, 0, 0, 0], 4, 5))['rotation'][1], 0)
        # A held keyboard enable cannot take over on disconnect.
        self.assertFalse(sample(True, {'SHIFT', 'W'}, None)['enabled'])
        sample(True, set(), None)
        self.assertTrue(sample(True, {'SHIFT', 'W'}, None)['enabled'])

    def test_standard_buttons_gripper_and_deadzone(self):
        buttons = [0] * 15
        pad = (1, [0., 0., 0., 0.], buttons)
        self.mapper.sample(True, set(), pad)
        buttons[4] = 1
        buttons[0] = 1
        result = self.mapper.sample(True, set(), (1, [.1, -.5, 0., 0.], buttons))
        self.assertEqual(result['direction'][0], 0)
        self.assertGreater(result['direction'][1], 0)
        self.assertEqual(result['gripper'], 1)
        buttons[1] = 1
        self.assertEqual(self.mapper.sample(True, set(), pad)['gripper'], 0)


if __name__ == '__main__':
    unittest.main()

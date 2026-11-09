import ctypes
import os
import unittest
from unittest.mock import MagicMock, patch

from scripts.native_input import InputMapper, NativeInput, RecordCommands, WindowsKeyboard, X11Keyboard


class NativeInputTest(unittest.TestCase):
    def setUp(self):
        self.mapper = InputMapper()

    def test_keyboard_and_pad_share_control_without_held_takeover(self):
        buttons = [0] * 15
        axes = [0.] * 4
        pad = (0, axes, buttons)
        sample = self.mapper.sample
        sample(True, set(), pad)
        axes[1] = -1.
        self.assertEqual(sample(True, {'SHIFT', 'A'}, pad)['direction'], [0, 1, 0])
        buttons[4] = 1
        self.assertEqual(sample(True, {'SHIFT', 'A'}, pad)['direction'], [0, 1, 0])
        self.assertFalse(sample(True, set(), pad)['enabled'])  # held LB cannot take over
        buttons[4] = 0
        sample(True, set(), pad)
        buttons[4] = 1
        self.assertGreater(sample(True, set(), pad)['direction'][0], 0)
        self.assertGreater(sample(True, {'SHIFT', 'A'}, pad)['direction'][0], 0)
        buttons[4] = 0
        self.assertFalse(sample(True, {'SHIFT', 'A'}, pad)['enabled'])
        sample(True, set(), pad)
        self.assertEqual(sample(True, {'SHIFT', 'A'}, pad)['direction'], [0, 1, 0])

    def test_focus_and_hotplug_require_release_of_each_deadman(self):
        buttons = [0] * 15
        pad = (0, [0., 0., 0., 0.], buttons)
        sample = self.mapper.sample
        sample(True, set(), None)
        sample(True, {'SHIFT', 'W'}, None)
        self.assertFalse(sample(True, {'SHIFT', 'W'}, pad)['enabled'])
        sample(True, set(), pad)
        self.assertTrue(sample(True, {'SHIFT', 'W'}, pad)['enabled'])
        sample(False, {'SHIFT', 'W'}, pad)
        self.assertFalse(sample(True, {'SHIFT', 'W'}, pad)['enabled'])
        sample(True, set(), pad)
        buttons[4] = 1
        self.assertTrue(sample(True, set(), pad)['enabled'])
        self.assertFalse(sample(True, {'SHIFT', 'W'}, None)['enabled'])
        sample(True, set(), None)
        self.assertTrue(sample(True, {'SHIFT', 'W'}, None)['enabled'])

    def test_manual_released_checks_all_gamepads_and_focus(self):
        inputs = NativeInput.__new__(NativeInput)
        inputs.keyboard = MagicMock()
        inputs.mapper = InputMapper()
        inputs.record_commands = RecordCommands()
        inputs._commands = []
        inputs.keyboard.read.return_value = (True, set())
        first, second = MagicMock(), MagicMock()
        first.axes, first.buttons = [0.] * 4, [0] * 15
        second.axes, second.buttons = [0.] * 4, [0] * 15
        with patch('scripts.native_input.glfw.poll_events'), \
             patch('scripts.native_input.glfw.joystick_present', side_effect=lambda i: i in (0, 1)), \
             patch('scripts.native_input.glfw.joystick_is_gamepad', return_value=True), \
             patch('scripts.native_input.glfw.get_gamepad_state', side_effect=lambda i: (first, second)[i]):
            inputs.poll()
            self.assertTrue(inputs.manual_released)
            second.buttons[4] = 1
            inputs.poll()
            self.assertFalse(inputs.manual_released)
            second.buttons[4] = 0
            inputs.keyboard.read.return_value = (True, {'SHIFT'})
            inputs.poll()
            self.assertFalse(inputs.manual_released)
            inputs.keyboard.read.return_value = (False, set())
            inputs.poll()
            self.assertFalse(inputs.manual_released)

    def test_record_keys_are_single_press_and_rearm_after_focus(self):
        commands = RecordCommands()
        self.assertEqual(commands.sample(True, set()), [])
        self.assertEqual(commands.sample(True, {'F8'}), ['start'])
        self.assertEqual(commands.sample(True, {'F8'}), [])
        self.assertEqual(commands.sample(False, {'F8'}), [])
        self.assertEqual(commands.sample(True, {'F8'}), [])
        self.assertEqual(commands.sample(True, set()), [])
        self.assertEqual(commands.sample(True, {'F8'}), ['start'])
        self.assertEqual(commands.sample(True, {'F9'}), ['success'])
        self.assertEqual(commands.sample(True, {'F10'}), ['failure'])
        self.assertEqual(commands.sample(True, {'F11'}), ['leader_toggle'])
        self.assertEqual(commands.sample(True, {'F11'}), [])
        self.assertEqual(commands.sample(False, {'F11'}), [])
        self.assertEqual(commands.sample(True, {'F11'}), [])
        self.assertEqual(commands.sample(True, set()), [])
        self.assertEqual(commands.sample(True, {'F11'}), ['leader_toggle'])

    def test_windows_reads_record_keys_while_focused(self):
        user = MagicMock()
        user.GetForegroundWindow.return_value = 101
        user.GetWindowThreadProcessId.side_effect = lambda _, process: setattr(
            ctypes.cast(process, ctypes.POINTER(ctypes.c_ulong)).contents, 'value', os.getpid())
        user.GetAsyncKeyState.side_effect = lambda key: 0x8000 if key in (0x77, 0x78, 0x79, 0x7a) else 0
        with patch.object(ctypes, 'WinDLL', return_value=user, create=True):
            focused, keys = WindowsKeyboard().read()
        self.assertTrue(focused)
        self.assertTrue({'F8', 'F9', 'F10', 'F11'}.issubset(keys))

    def test_x11_reads_f11_while_focused(self):
        keyboard = X11Keyboard.__new__(X11Keyboard)
        keyboard.x = MagicMock()
        keyboard.display, keyboard.atom = 1, 1
        keyboard._owned = lambda _: True
        keyboard.x.XKeysymToKeycode.side_effect = lambda _, keysym: 42 if keysym == 0xffc8 else 0
        def set_keymap(_, keymap):
            data = bytearray(32)
            data[42 // 8] |= 1 << (42 % 8)
            ctypes.memmove(keymap, bytes(data), len(data))
        keyboard.x.XQueryKeymap.side_effect = set_keymap
        focused, keys = keyboard.read()
        self.assertTrue(focused)
        self.assertIn('F11', keys)

    def test_poll_exposes_record_commands_and_drops_them_on_focus_loss(self):
        inputs = NativeInput.__new__(NativeInput)
        inputs.keyboard = MagicMock()
        inputs.mapper = InputMapper()
        inputs.record_commands = RecordCommands()
        inputs._commands = []
        samples = iter([(True, set()), (True, {'F8'}), (True, {'F8'}),
                        (True, set()), (True, {'F9'}), (False, set())])
        inputs.keyboard.read.side_effect = lambda: next(samples)
        with patch('scripts.native_input.glfw.poll_events'), \
             patch('scripts.native_input.glfw.joystick_present', return_value=False):
            inputs.poll()
            inputs.poll()
            self.assertEqual(inputs.pop_commands(), ['start'])
            inputs.poll()
            self.assertEqual(inputs.pop_commands(), [])
            inputs.poll()
            inputs.poll()
            inputs.poll()  # Focus loss clears an unconsumed event.
            self.assertEqual(inputs.pop_commands(), [])

    def test_keyboard_requires_release_after_start_and_focus_loss(self):
        sample = self.mapper.sample
        self.assertFalse(sample(True, {'SHIFT', 'W'}, None)['enabled'])
        self.assertFalse(sample(True, set(), None)['enabled'])
        self.assertEqual(sample(True, {'SHIFT', 'W'}, None)['direction'], [1, 0, 0])
        self.assertFalse(sample(False, {'SHIFT', 'W'}, None)['enabled'])
        self.assertFalse(sample(True, {'SHIFT', 'W'}, None)['enabled'])
        sample(True, set(), None)
        self.assertTrue(sample(True, {'SHIFT', 'W'}, None)['enabled'])
        self.assertFalse(sample(True, {'W'}, None)['enabled'])

    def test_windows_focus_accepts_both_viewers_and_rearms_after_leaving(self):
        user = MagicMock()
        foreground = [101]
        pressed = set()
        user.GetForegroundWindow.side_effect = lambda: foreground[0]

        def window_process(window, process):
            pid = os.getpid() if window in (101, 102) else os.getpid() + 1
            ctypes.cast(process, ctypes.POINTER(ctypes.c_ulong))[0] = pid

        user.GetWindowThreadProcessId.side_effect = window_process
        user.GetAsyncKeyState.side_effect = lambda key: 0x8000 if key in pressed else 0
        with patch.object(ctypes, 'WinDLL', return_value=user, create=True):
            keyboard = WindowsKeyboard()
            sample = lambda: self.mapper.sample(*keyboard.read(), None)
            self.assertFalse(sample()['enabled'])
            pressed.update((0x10, ord('W')))
            self.assertEqual(sample()['direction'], [1, 0, 0])
            foreground[0] = 102  # The other viewer belongs to the same process.
            self.assertEqual(sample()['direction'], [1, 0, 0])
            foreground[0] = 103  # Another application disables input.
            self.assertFalse(sample()['enabled'])
            foreground[0] = 101
            self.assertFalse(sample()['enabled'])  # Held Shift must be released.
            pressed.clear()
            self.assertFalse(sample()['enabled'])
            pressed.update((0x10, ord('W')))
            self.assertEqual(sample()['direction'], [1, 0, 0])

            # Gamepad input follows the same focus and release rules.
            mapper = InputMapper()
            buttons, axes = [0] * 15, [0.] * 4
            def gamepad_sample():
                focused, keys = keyboard.read()
                return mapper.sample(focused, keys, (0, axes, buttons))
            self.assertFalse(gamepad_sample()['enabled'])
            buttons[4], axes[1] = 1, -1.
            self.assertGreater(gamepad_sample()['direction'][0], 0)
            foreground[0] = 102
            self.assertGreater(gamepad_sample()['direction'][0], 0)
            foreground[0] = 103
            self.assertFalse(gamepad_sample()['enabled'])
            foreground[0] = 102
            self.assertFalse(gamepad_sample()['enabled'])
            buttons[4], axes[1] = 0, 0.
            self.assertFalse(gamepad_sample()['enabled'])
            buttons[4], axes[1] = 1, -1.
            self.assertGreater(gamepad_sample()['direction'][0], 0)

    def test_gamepad_axis_recenter_mode_and_disconnect(self):
        sample = self.mapper.sample
        def pad(axes, *pressed):
            buttons = [int(i in pressed) for i in range(15)]
            return (0, axes, buttons)
        centered = [0., 0., 0., 0.]
        sample(True, set(), pad(centered))
        self.assertLess(sample(True, set(), pad([.5, 0, 0, 0], 4))['direction'][1], 0)
        self.assertEqual(sample(True, set(), pad([.5, 0, 0, 0], 4, 5))['rotation'], [0, 0, 0])
        sample(True, set(), pad(centered, 4, 5))
        self.assertLess(sample(True, set(), pad([.5, 0, 0, 0], 4, 5))['rotation'][1], 0)
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
        self.assertEqual(result['direction'][1], 0)
        self.assertGreater(result['direction'][0], 0)
        self.assertEqual(result['gripper'], 1)
        buttons[1] = 1
        self.assertEqual(self.mapper.sample(True, set(), pad)['gripper'], 0)

    def test_keyboard_axes_match_world_frame(self):
        self.mapper.sample(True, set(), None)
        for key, field, expected in [
            ('W', 'direction', [1, 0, 0]), ('S', 'direction', [-1, 0, 0]),
            ('A', 'direction', [0, 1, 0]), ('D', 'direction', [0, -1, 0]),
            ('Q', 'direction', [0, 0, 1]), ('E', 'direction', [0, 0, -1]),
            ('I', 'rotation', [1, 0, 0]), ('K', 'rotation', [-1, 0, 0]),
            ('J', 'rotation', [0, 1, 0]), ('L', 'rotation', [0, -1, 0]),
            ('U', 'rotation', [0, 0, 1]), ('O', 'rotation', [0, 0, -1]),
        ]:
            with self.subTest(key=key):
                result = self.mapper.sample(True, {'SHIFT', key}, None)
                self.assertEqual(result[field], expected)
                other = 'rotation' if field == 'direction' else 'direction'
                self.assertEqual(result[other], [0, 0, 0])

    def test_gamepad_primary_axis_rejects_diagonal_noise(self):
        def pad(axes, rotation=False):
            buttons = [0] * 15
            buttons[4], buttons[5] = 1, int(rotation)
            return (0, axes, buttons)
        buttons = [0] * 15
        self.mapper.sample(True, set(), (0, [0.] * 4, buttons))
        self.assertEqual(self.mapper.sample(True, set(), pad([0.] * 4))['direction'], [0, 0, 0])
        # Left stick tilted mostly forward: no Y or Z drift despite noise.
        result = self.mapper.sample(True, set(), pad([.3, -.8, 0., .2]))
        self.assertGreater(result['direction'][0], 0)
        self.assertEqual(result['direction'][1:], [0, 0])
        # Switch modes with centered sticks, then use right-stick X for Z rotation.
        self.mapper.sample(True, set(), pad([0.] * 4, True))
        result = self.mapper.sample(True, set(), pad([.2, -.3, .8, .9], True))
        self.assertEqual(result['rotation'][:2], [0, 0])
        self.assertGreater(result['rotation'][2], 0)
        self.assertEqual(result['direction'], [0, 0, 0])

    def test_gamepad_translation_axes_follow_world_frame(self):
        buttons = [0] * 15
        self.mapper.sample(True, set(), (0, [0.] * 4, buttons))
        buttons[4] = 1
        for raw, world_index, sign in [
            ([0, -1, 0, 0], 0, 1), ([0, 1, 0, 0], 0, -1),
            ([-1, 0, 0, 0], 1, 1), ([1, 0, 0, 0], 1, -1),
            ([0, 0, 0, -1], 2, 1), ([0, 0, 0, 1], 2, -1),
        ]:
            with self.subTest(raw=raw):
                direction = self.mapper.sample(True, set(), (0, raw, buttons))['direction']
                expected = [0, 0, 0]
                expected[world_index] = sign
                self.assertEqual(direction, expected)

    def test_gamepad_rotation_axes_are_independent_in_world_frame(self):
        buttons = [0] * 15
        self.mapper.sample(True, set(), (0, [0.] * 4, buttons))
        buttons[4] = buttons[5] = 1
        self.mapper.sample(True, set(), (0, [0.] * 4, buttons))
        for raw, world_index, sign in [
            ([0, -1, 0, 0], 0, 1), ([0, 1, 0, 0], 0, -1),
            ([-1, 0, 0, 0], 1, 1), ([1, 0, 0, 0], 1, -1),
            ([0, 0, 1, 0], 2, 1), ([0, 0, -1, 0], 2, -1),
        ]:
            with self.subTest(raw=raw):
                rotation = self.mapper.sample(True, set(), (0, raw, buttons))['rotation']
                expected = [0, 0, 0]
                expected[world_index] = sign
                self.assertEqual(rotation, expected)

    def test_rotation_axis_does_not_flicker_near_diagonal(self):
        buttons = [0] * 15
        self.mapper.sample(True, set(), (0, [0.] * 4, buttons))
        buttons[4] = buttons[5] = 1
        self.mapper.sample(True, set(), (0, [0.] * 4, buttons))
        sample = lambda axes: self.mapper.sample(True, set(), (0, axes, buttons))['rotation']

        self.assertGreater(sample([-.68, -.70, 0, 0])[0], 0)
        # Small fluctuations around equal deflection must not change axes.
        for axes in ([-.71, -.69, 0, 0], [-.69, -.71, 0, 0]):
            result = sample(axes)
            self.assertGreater(result[0], 0)
            self.assertEqual(result[1:], [0, 0])
        # A clearly deliberate move to another axis still switches normally.
        result = sample([-.9, -.4, 0, 0])
        self.assertEqual(result[0], 0)
        self.assertGreater(result[1], 0)
        # Releasing the selected axis allows a new axis immediately.
        result = sample([0, -.2, 0, 0])
        self.assertGreater(result[0], 0)
        self.assertEqual(result[1], 0)
        self.assertEqual(sample([0, 0, 0, 0]), [0, 0, 0])

    def test_rotation_axis_resets_after_deadman_release(self):
        buttons = [0] * 15
        self.mapper.sample(True, set(), (0, [0.] * 4, buttons))
        buttons[4] = buttons[5] = 1
        self.mapper.sample(True, set(), (0, [0.] * 4, buttons))
        self.mapper.sample(True, set(), (0, [-.68, -.70, 0, 0], buttons))
        buttons[4] = 0
        self.assertFalse(self.mapper.sample(True, set(), (0, [0.] * 4, buttons))['enabled'])
        buttons[4] = 1
        result = self.mapper.sample(True, set(), (0, [-.71, -.69, 0, 0], buttons))['rotation']
        self.assertEqual(result[0], 0)
        self.assertGreater(result[1], 0)


if __name__ == '__main__':
    unittest.main()

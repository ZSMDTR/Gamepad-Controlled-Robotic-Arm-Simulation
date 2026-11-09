"""Inputs for the MuJoCo passive viewer, without a web browser.

Read-only keyboard/gamepad polling; the caller must send a disabled sample on
exit. Unsupported displays fail closed instead of assuming focus.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import sys

import glfw


def disabled():
    return dict(direction=[0., 0., 0.], rotation=[0., 0., 0.], gripper=0, enabled=False)


def axis(value):
    value = max(-1., min(1., float(value)))
    return math.copysign((abs(value) - .12) / .88, value) if abs(value) > .12 else 0.


def unit(values):
    length = math.sqrt(sum(value * value for value in values))
    return [value / max(1., length) for value in values]


def primary_axis(values):
    """Keep the strongest stick axis so a slightly diagonal push stays axial."""
    index = max(range(len(values)), key=lambda i: abs(values[i]))
    return [value if i == index else 0. for i, value in enumerate(values)]


class RecordCommands:
    """Emit one recording command per press while a viewer has focus."""
    COMMANDS = (('F8', 'start'), ('F9', 'success'), ('F10', 'failure'),
                ('F11', 'leader_toggle'), ('F12', 'leader_save_home'))

    def __init__(self):
        self.focused = False
        self.previous = set()

    def sample(self, focused, keys):
        down = {key for key, _ in self.COMMANDS if key in keys}
        if not focused:
            self.focused = False
            self.previous = set()
            return []
        # Keys held while a viewer gains focus cannot trigger an action.
        if not self.focused:
            self.focused = True
            self.previous = down
            return []
        commands = [command for key, command in self.COMMANDS
                    if key in down and key not in self.previous]
        self.previous = down
        return commands


class _SingleInputMapper:
    """Fail-closed state for one keyboard or gamepad source."""
    def __init__(self):
        self.source = None
        self.needs_release = True
        self.mode = None
        self.blocked = [False] * 4
        self.rotation_axis = None

    def rotation_primary_axis(self, values):
        """Keep the selected axis until another is clearly stronger."""
        strongest = max(range(len(values)), key=lambda i: abs(values[i]))
        current = self.rotation_axis
        if not any(values):
            self.rotation_axis = None
            return [0.] * len(values)
        if current is None or not values[current] or abs(values[strongest]) > abs(values[current]) + .15:
            current = strongest
        self.rotation_axis = current
        return [value if i == current else 0. for i, value in enumerate(values)]

    def sample(self, focused, keys, pad):
        # pad is (stable identity, axes, buttons) or None. A GLFW standard
        # mapping uses browser-compatible face and shoulder button indices.
        source = ('pad', pad[0]) if pad else ('keyboard',)
        if not focused:
            self.source = None
            self.needs_release = True
            self.mode = None
            self.rotation_axis = None
            return disabled()
        if source != self.source:
            self.source = source
            self.needs_release = True
            self.mode = None
            self.rotation_axis = None
        if pad:
            _, raw_axes, buttons = pad
            axes = [axis(raw_axes[i]) for i in range(4)]
            pressed = bool(buttons[4])
            mode = 'rotation' if buttons[5] else 'translation'
            if mode != self.mode:
                self.mode = mode
                self.blocked = [value != 0 for value in axes]
                self.rotation_axis = None
            for i, value in enumerate(axes):
                if value == 0:
                    self.blocked[i] = False
            axes = [0. if self.blocked[i] else value for i, value in enumerate(axes)]
            if not pressed:
                self.needs_release = False
            if not pressed or self.needs_release:
                self.rotation_axis = None
                return disabled()
            # World frame: +X forward from base, +Y robot left, +Z up.
            # In rotation mode the inputs are independent world-axis angular
            # commands; no joystick axis contributes to two rotations.
            direction = primary_axis([-axes[1], -axes[0], -axes[3]]) if mode == 'translation' else [0., 0., 0.]
            rotation = self.rotation_primary_axis([-axes[1], -axes[0], axes[2]]) if mode == 'rotation' else [0., 0., 0.]
            return dict(direction=direction, rotation=rotation,
                        gripper=int(bool(buttons[0])) - int(bool(buttons[1])), enabled=True)
        pressed = 'SHIFT' in keys
        if not pressed:
            self.needs_release = False
        if not pressed or self.needs_release:
            return disabled()
        def pair(a, b):
            return float(a in keys) - float(b in keys)
        return dict(direction=unit([pair('W', 'S'), pair('A', 'D'), pair('Q', 'E')]),
                    rotation=unit([pair('I', 'K'), pair('J', 'L'), pair('U', 'O')]),
                    gripper=int('Z' in keys) - int('X' in keys), enabled=True)


class InputMapper:
    """Arbitrate two independent deadmen without changing a held source."""
    def __init__(self):
        self.keyboard = _SingleInputMapper()
        self.gamepad = _SingleInputMapper()
        self.pad_id = None
        self.owner = None
        self.blocked = set()
        self.all_released = True

    def sample(self, focused, keys, pad):
        pad_id = pad[0] if pad else None
        keyboard_down = 'SHIFT' in keys
        pad_down = bool(pad and pad[2][4])
        self.all_released = not keyboard_down and not pad_down
        if not focused or pad_id != self.pad_id:
            # Disconnect, replacement, focus loss, and hot-plug invalidate both
            # deadmen. Neither a held Shift nor LB may take over unexpectedly.
            self.keyboard.sample(False, set(), None)
            self.gamepad.sample(False, set(), None)
            self.owner = None
            self.blocked.clear()
            self.pad_id = pad_id
        keyboard = self.keyboard.sample(focused, keys, None)
        gamepad = self.gamepad.sample(focused and pad is not None, set(), pad)
        if not focused:
            return disabled()
        if not keyboard_down:
            self.blocked.discard('keyboard')
        if not pad_down:
            self.blocked.discard('gamepad')
        samples = {'keyboard': keyboard, 'gamepad': gamepad}
        down = {'keyboard': keyboard_down, 'gamepad': pad_down}
        if self.owner is not None and not samples[self.owner]['enabled']:
            self.owner = None
        if self.owner is None:
            # Simultaneous fresh presses select the gamepad deterministically.
            for source in ('gamepad', 'keyboard'):
                if samples[source]['enabled'] and source not in self.blocked:
                    self.owner = source
                    break
        if self.owner is None:
            return disabled()
        other = 'keyboard' if self.owner == 'gamepad' else 'gamepad'
        if down[other]:
            self.blocked.add(other)
        return samples[self.owner]


class WindowsKeyboard:
    def __init__(self):
        self.user = ctypes.WinDLL('user32', use_last_error=True)
        self.user.GetForegroundWindow.restype = ctypes.c_void_p
        self.user.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        self.user.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user.GetAsyncKeyState.restype = ctypes.c_short

    def read(self):
        hwnd = self.user.GetForegroundWindow()
        pid = ctypes.c_ulong()
        self.user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not hwnd or pid.value != os.getpid():
            return False, set()
        keys = {key for key in 'WASDQEIKLJOUZX' if self.user.GetAsyncKeyState(ord(key)) & 0x8000}
        keys.update(name for name, code in (('F8', 0x77), ('F9', 0x78), ('F10', 0x79), ('F11', 0x7a), ('F12', 0x7b))
                    if self.user.GetAsyncKeyState(code) & 0x8000)
        if self.user.GetAsyncKeyState(0x10) & 0x8000:
            keys.add('SHIFT')
        return True, keys

    def close(self):
        pass


class X11Keyboard:
    """X11 focus ownership and key state; unavailable on native Wayland."""
    def __init__(self):
        lib = ctypes.util.find_library('X11')
        if not lib:
            raise RuntimeError('X11 library not found')
        self.x = ctypes.CDLL(lib)
        p, ul, i = ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int
        for name, args, result in [
            ('XOpenDisplay', [ctypes.c_char_p], p), ('XCloseDisplay', [p], i),
            ('XGetInputFocus', [p, ctypes.POINTER(ul), ctypes.POINTER(i)], i),
            ('XInternAtom', [p, ctypes.c_char_p, i], ul),
            ('XGetWindowProperty', [p, ul, ul, ctypes.c_long, ctypes.c_long, i, ul, ctypes.POINTER(ul), ctypes.POINTER(i), ctypes.POINTER(ul), ctypes.POINTER(ul), ctypes.POINTER(p)], i),
            ('XQueryTree', [p, ul, ctypes.POINTER(ul), ctypes.POINTER(ul), ctypes.POINTER(p), ctypes.POINTER(ctypes.c_uint)], i),
            ('XQueryKeymap', [p, ctypes.c_void_p], i), ('XKeysymToKeycode', [p, ul], ctypes.c_ubyte),
            ('XFree', [p], i),
        ]:
            function = getattr(self.x, name)
            function.argtypes, function.restype = args, result
        self.display = self.x.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError('X11 display unavailable (native Wayland is not supported)')
        self.atom = self.x.XInternAtom(self.display, b'_NET_WM_PID', 1)

    def _owned(self, window):
        # Focus may be on a child widget; check its ancestors for the PID.
        for _ in range(16):
            if window <= 1:
                break
            actual, fmt = ctypes.c_ulong(), ctypes.c_int()
            count, remaining = ctypes.c_ulong(), ctypes.c_ulong()
            value = ctypes.c_void_p()
            self.x.XGetWindowProperty(self.display, window, self.atom, 0, 1, 0, 0,
                                      ctypes.byref(actual), ctypes.byref(fmt), ctypes.byref(count),
                                      ctypes.byref(remaining), ctypes.byref(value))
            if value:
                try:
                    if fmt.value == 32 and count.value and ctypes.cast(value, ctypes.POINTER(ctypes.c_ulong))[0] == os.getpid():
                        return True
                finally:
                    self.x.XFree(value)
            root, parent, children, n = ctypes.c_ulong(), ctypes.c_ulong(), ctypes.c_void_p(), ctypes.c_uint()
            ok = self.x.XQueryTree(self.display, window, ctypes.byref(root), ctypes.byref(parent), ctypes.byref(children), ctypes.byref(n))
            if children:
                self.x.XFree(children)
            if not ok or parent.value == window:
                break
            window = parent.value
        return False

    def read(self):
        window, revert = ctypes.c_ulong(), ctypes.c_int()
        self.x.XGetInputFocus(self.display, ctypes.byref(window), ctypes.byref(revert))
        if not self.atom or not self._owned(window.value):
            return False, set()
        keymap = ctypes.create_string_buffer(32)
        self.x.XQueryKeymap(self.display, keymap)
        def down(keysym):
            code = self.x.XKeysymToKeycode(self.display, keysym)
            return bool(code and keymap.raw[code // 8] & (1 << (code % 8)))
        keys = {key for key in 'WASDQEIKLJOUZX' if down(ord(key.lower()))}
        keys.update(name for name, keysym in (('F8', 0xffc5), ('F9', 0xffc6), ('F10', 0xffc7), ('F11', 0xffc8), ('F12', 0xffc9))
                    if down(keysym))
        if down(0xffe1) or down(0xffe2):
            keys.add('SHIFT')
        return True, keys

    def close(self):
        self.x.XCloseDisplay(self.display)


class NativeInput:
    """Poll from the caller's viewer loop and pass result to Simulation.set_input."""
    def __init__(self):
        if sys.platform == 'win32':
            self.keyboard = WindowsKeyboard()
        elif sys.platform.startswith('linux'):
            self.keyboard = X11Keyboard()
        else:
            raise RuntimeError('native input supports Windows and Linux X11 only')
        if not glfw.init():
            self.keyboard.close()
            raise RuntimeError('GLFW gamepad initialization failed')
        self.mapper = InputMapper()
        self.record_commands = RecordCommands()
        self._commands = []
        self.manual_released = False

    def poll(self):
        # Process joystick hot-plug notifications on the initializing thread.
        glfw.poll_events()
        focused, keys = self.keyboard.read()
        if not focused:
            self._commands.clear()
        self._commands.extend(self.record_commands.sample(focused, keys))
        pad = None
        any_pad_deadman = False
        for joystick in range(glfw.JOYSTICK_1, glfw.JOYSTICK_LAST + 1):
            if glfw.joystick_present(joystick) and glfw.joystick_is_gamepad(joystick):
                state = glfw.get_gamepad_state(joystick)
                if state:
                    any_pad_deadman |= bool(state.buttons[4])
                    if pad is None:
                        pad = (joystick, state.axes, state.buttons)
        self.manual_released = focused and 'SHIFT' not in keys and not any_pad_deadman
        return self.mapper.sample(focused, keys, pad)

    def pop_commands(self):
        """Return and clear commands; call from the same thread as poll()."""
        commands, self._commands = self._commands, []
        return commands

    def close(self):
        self.keyboard.close()
        # The MuJoCo viewer closes asynchronously and also owns GLFW. Its
        # registered process-exit cleanup must run after its windows finish.

"""Place this process's two Windows MuJoCo viewers side by side."""
import ctypes
from ctypes import wintypes
import os
import sys


def split_work_area(left, top, right, bottom):
    """Return two window rectangles inside a monitor's usable work area."""
    middle = left + (right - left) // 2
    return ((left, top, middle - left, bottom - top),
            (middle, top, right - middle, bottom - top))


def viewer_windows():
    """Return visible MuJoCo window handles belonging to this process."""
    if sys.platform != 'win32':
        return set()
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    windows = set()

    @callback_type
    def visit(handle, _):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        if pid.value == os.getpid() and user32.IsWindowVisible(handle):
            title = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(handle, title, len(title))
            if title.value.startswith('MuJoCo'):
                windows.add(handle)
        return True

    user32.EnumWindows(visit, 0)
    return windows


def arrange_viewers(main_windows, current_windows):
    """Position just the newly created viewer and its single main viewer.

    On another OS, or if identifying either window is ambiguous, do nothing.
    """
    if sys.platform != 'win32' or len(main_windows) != 1:
        return False
    added = current_windows - main_windows
    if len(added) != 1 or not main_windows <= current_windows:
        return False
    main, auxiliary = next(iter(main_windows)), next(iter(added))
    user32 = ctypes.WinDLL('user32', use_last_error=True)

    class MonitorInfo(ctypes.Structure):
        _fields_ = [('cbSize', wintypes.DWORD), ('rcMonitor', wintypes.RECT),
                    ('rcWork', wintypes.RECT), ('dwFlags', wintypes.DWORD)]

    user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.MonitorFromWindow.restype = wintypes.HANDLE
    user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MonitorInfo)]
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
    monitor = user32.MonitorFromWindow(main, 2)  # nearest monitor
    info = MonitorInfo()
    info.cbSize = ctypes.sizeof(info)
    if not monitor or not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
        return False
    work = info.rcWork
    rectangles = split_work_area(work.left, work.top, work.right, work.bottom)
    # Preserve stacking order and focus; the windows remain freely resizable.
    return all(user32.SetWindowPos(handle, None, *rectangle, 0x0004 | 0x0010)
               for handle, rectangle in zip((main, auxiliary), rectangles))

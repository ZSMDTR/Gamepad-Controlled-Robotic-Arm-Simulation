import unittest
from unittest.mock import patch

from scripts.window_layout import arrange_viewers, split_work_area


class WindowLayoutTest(unittest.TestCase):
    def test_split_uses_full_work_area_even_on_left_monitor(self):
        self.assertEqual(split_work_area(-1921, 40, -1, 1040),
                         ((-1921, 40, 960, 1000), (-961, 40, 960, 1000)))
        self.assertEqual(split_work_area(0, 0, 1707, 1019),
                         ((0, 0, 853, 1019), (853, 0, 854, 1019)))

    def test_ambiguous_windows_are_not_moved(self):
        with patch('scripts.window_layout.sys.platform', 'win32'):
            self.assertFalse(arrange_viewers(set(), {1}))
            self.assertFalse(arrange_viewers({1, 2}, {1, 2, 3}))
            self.assertFalse(arrange_viewers({1}, {1, 2, 3}))
            self.assertFalse(arrange_viewers({1}, {2}))

    def test_other_platforms_leave_layout_to_window_manager(self):
        with patch('scripts.window_layout.sys.platform', 'linux'):
            self.assertFalse(arrange_viewers({1}, {1, 2}))


if __name__ == '__main__':
    unittest.main()

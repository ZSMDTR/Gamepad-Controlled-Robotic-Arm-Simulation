import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mujoco

from arm_teleop.leader_home import load_home, save_home
from arm_teleop.server import Simulation
from scripts.native_viewer import LeaderHomeCapture, show_native_viewer, start_leader_at_home


def sample(values, timestamp):
    return SimpleNamespace(positions=tuple(values), timestamp=timestamp)


class NativeHomeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'leader-home.json'
        self.reader = SimpleNamespace(value=None)
        self.reader.latest = lambda: self.reader.value

    def test_capture_waits_for_stable_samples_and_uses_only_new_data(self):
        capture = LeaderHomeCapture(self.path)
        capture.begin(1.)
        for t in (0.99, 1.00, 1.08, 1.16, 1.24):
            self.reader.value = sample([0.] * 7, t)
            self.assertIsNone(capture.update(self.reader, max(t, 1.)))
        self.assertFalse(self.path.exists())
        self.reader.value = sample([0.] * 7, 1.32)
        self.assertEqual(capture.update(self.reader, 1.32), 'HOME SAVED - F11')
        self.assertEqual(load_home(self.path)['positions_rad'], [0.] * 7)

    def test_motion_gap_stale_and_focus_cancel_leave_old_reference_unchanged(self):
        samples = [sample([0.] * 7, t) for t in (0., .1, .2, .3)]
        old = save_home(self.path, samples, .3)
        capture = LeaderHomeCapture(self.path)
        capture.begin(1.)
        for t, value in ((1., 0.), (1.1, .2), (1.2, 0.), (1.3, .2)):
            self.reader.value = sample([value] * 7, t)
            self.assertIsNone(capture.update(self.reader, t))
        capture.cancel()
        self.assertEqual(load_home(self.path), old)
        capture.begin(2.)
        self.reader.value = sample([0.] * 7, 1.)
        self.assertEqual(capture.update(self.reader, 2.), 'HOME NO DATA - F12')
        self.assertEqual(load_home(self.path), old)

    def test_missing_wrong_pose_and_stale_are_rejected_and_home_starts(self):
        sim = MagicMock()
        sim.start_leader.return_value = True
        self.reader.value = sample([0.] * 7, 1.)
        self.assertFalse(start_leader_at_home(sim, self.reader, {}, self.path, 1.))
        self.assertEqual(sim.leader_status, 'HOME NOT SET - F12')
        save_home(self.path, [sample([0.] * 7, t) for t in (.6, .7, .8, .9, 1.)], 1.)
        self.reader.value = sample([0.] * 6 + [.2], 1.)
        self.assertFalse(start_leader_at_home(sim, self.reader, {}, self.path, 1.))
        self.assertEqual(sim.leader_status, 'NOT HOME - IDs 6')
        self.reader.value = sample([0.] * 7, .5)
        self.assertFalse(start_leader_at_home(sim, self.reader, {}, self.path, 1.))
        self.assertEqual(sim.leader_status, 'NO FRESH DATA')
        self.reader.value = sample([0.] * 7, 1.)
        self.assertTrue(start_leader_at_home(sim, self.reader, {}, self.path, 1.))
        sim.start_leader.assert_called_once()

    def test_viewer_cancels_on_focus_loss_and_blocks_save_while_recording(self):
        sim = Simulation()
        loop = {'n': 0}
        events = []

        class Inputs:
            record_commands = SimpleNamespace(focused=True)
            manual_released = True
            def poll(self):
                loop['n'] += 1
                self.record_commands.focused = loop['n'] != 2
                return {'direction': [0., 0., 0.], 'enabled': False}
            def pop_commands(self):
                return {1: ['leader_save_home'], 3: ['start'], 4: ['leader_save_home']}.get(loop['n'], [])
            def close(self): pass

        class Viewer:
            user_scn = mujoco.MjvScene(sim.backend.model, maxgeom=50)
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def is_running(self): return loop['n'] < 4
            def sync(self): pass

        class Reader:
            def latest(self): return sample([0.] * 7, loop['n'] * .1)

        recorder = SimpleNamespace(path='mock', steps=1, finish=lambda *args: None)
        clock = SimpleNamespace(monotonic=lambda: loop['n'] * .1, sleep=lambda _: None)
        def tick(*_): events.append(sim.leader_status)
        with patch('scripts.native_viewer.time', clock), \
             patch('scripts.native_viewer.NativeInput', Inputs), \
             patch('scripts.native_viewer.LEADER_HOME_PATH', self.path), \
             patch('scripts.native_viewer.EpisodeRecorder', return_value=recorder), \
             patch('scripts.native_viewer.prepare_auxiliary_view', return_value=(None, None, None, None)), \
             patch('scripts.native_viewer.viewer_windows', return_value=set()), \
             patch('scripts.native_viewer.mujoco.viewer.launch_passive', return_value=Viewer()), \
             patch.object(sim, 'tick', side_effect=tick):
            show_native_viewer(sim, Reader(), {})
        self.assertEqual(events, ['HOLD STILL - F12', 'HOME CANCELLED - F12',
                                  'HOME CANCELLED - F12', 'STOP RECORDING - F12'])
        self.assertFalse(self.path.exists())


if __name__ == '__main__':
    unittest.main()

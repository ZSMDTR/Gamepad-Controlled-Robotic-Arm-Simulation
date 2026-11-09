"""Leader following stays relative, bounded and fail-closed."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from arm_teleop.recorder import EpisodeRecorder, recording_metadata
from arm_teleop.server import DT, Simulation, leader_configuration
from scripts.native_viewer import show_native_viewer
from scripts.replay_episode import _validate_action, inspect_episode
from unittest.mock import patch

import mujoco


CONFIG = {'joint_indices': list(range(6)), 'joint_scales': [1., 1., -1., -1., -1., 1.],
          'gripper_scale': 1.2}


def sample(positions, timestamp):
    return SimpleNamespace(positions=positions, timestamp=timestamp)


class LeaderFollowTest(unittest.TestCase):
    def setUp(self):
        self.sim = Simulation()
        self.now = time.monotonic()
        self.sim.set_input({'direction': [0., 0., 0.], 'enabled': False})

    def tick(self):
        self.sim.tick(self.now, self.now)

    def test_relative_alignment_velocity_and_gripper_limits(self):
        sim = self.sim
        original = sim.backend.state()
        self.assertTrue(sim.start_leader(sample([1.] * 7, self.now), CONFIG, self.now))
        self.tick()
        np.testing.assert_allclose(sim.leader_ref, original['q'], atol=1e-10)
        self.now += DT
        sim.set_leader_sample(sample([1.2] * 7, self.now), self.now)
        before_ref = sim.leader_ref.copy()
        before_gripper = sim.gripper_target
        self.tick()
        self.assertGreater(sim.leader_ref[0], before_ref[0])
        self.assertLess(sim.leader_ref[2], before_ref[2])
        self.assertTrue(np.all(np.abs(sim.leader_ref - before_ref) <= .1 * DT + 1e-10))
        self.assertLessEqual(sim.gripper_target - before_gripper, .2 * DT + 1e-10)
        self.assertGreaterEqual(sim.gripper_target, 0.)
        self.assertLessEqual(sim.gripper_target, 1.)

    def test_wrap_and_jump_latch(self):
        sim = self.sim
        origin = [3.13] + [0.] * 6
        self.assertTrue(sim.start_leader(sample(origin, self.now), CONFIG, self.now))
        self.now += DT
        wrapped = [-3.13] + [0.] * 6
        sim.set_leader_sample(sample(wrapped, self.now), self.now)
        self.assertAlmostEqual(sim.leader_unwrapped[0] - 3.13, 2 * np.pi - 6.26)
        self.tick()
        self.assertTrue(sim.leader_active)
        self.now += DT
        jumped = [-2.] + [0.] * 6
        sim.set_leader_sample(sample(jumped, self.now), self.now)
        self.tick()
        self.assertFalse(sim.leader_active)
        self.assertIn('JUMP', sim.leader_status)
        self.now += DT
        sim.set_leader_sample(sample(jumped, self.now), self.now)
        self.tick()
        self.assertFalse(sim.leader_active)
        self.assertTrue(sim.start_leader(sample(jumped, self.now), CONFIG, self.now))
        np.testing.assert_allclose(sim.leader_ref, sim.backend.state()['q'])

    def test_stale_input_stops_and_does_not_restart(self):
        sim = self.sim
        self.assertTrue(sim.start_leader(sample([0.] * 7, self.now), CONFIG, self.now))
        self.now += .51
        self.tick()
        self.assertFalse(sim.leader_active)
        self.assertIn('STALE', sim.leader_status)
        sim.set_leader_sample(sample([.1] * 7, self.now), self.now)
        self.tick()
        self.assertFalse(sim.leader_active)

    def test_recording_identifies_source_and_preserves_raw_sample(self):
        sim = self.sim
        values = [0., .1, .2, .3, .4, .5, .6]
        with tempfile.TemporaryDirectory() as directory:
            metadata = recording_metadata(sim, 'robot_only', DT)
            metadata['leader_calibration'] = CONFIG
            sim.recorder = EpisodeRecorder(Path(directory), metadata)
            path = sim.recorder.path
            self.assertTrue(sim.start_leader(sample(values, self.now), CONFIG, self.now))
            self.tick()
            sim.recorder.finish('success', 'user_label')
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            inspect_episode(path)
        action = rows[1]['action']
        self.assertEqual(action['source'], 'leader')
        self.assertTrue(action['enabled'])
        self.assertEqual(action['leader_sample']['positions'], values)
        self.assertEqual(action['leader_sample']['timestamp'], self.now)
        self.assertEqual(len(action['q_ref']), 6)

    def test_invalid_mapping_is_rejected_before_use(self):
        for bad in (
            dict(CONFIG, joint_indices=[0] * 6),
            dict(CONFIG, joint_indices=['0', 1, 2, 3, 4, 5]),
            dict(CONFIG, joint_scales=[1., 1., float('nan'), 1., 1., 1.]),
            dict(CONFIG, gripper_scale=3.),
        ):
            with self.subTest(config=bad), self.assertRaises(ValueError):
                leader_configuration(bad)

    def test_replay_rejects_bad_leader_sample_but_accepts_legacy(self):
        legacy = {'direction': [0.] * 3, 'rotation': [0.] * 3, 'enabled': True, 'gripper': 0}
        _validate_action(legacy)
        with self.assertRaises(ValueError):
            _validate_action(dict(legacy, source='leader', leader_sample={'positions': [0.] * 6, 'timestamp': 1.}))

    def test_viewer_focus_loss_stops_leader_and_suppresses_held_manual_input(self):
        sim = self.sim
        events = []
        count = {'n': 0}

        class Inputs:
            record_commands = SimpleNamespace(focused=True)
            manual_released = False
            def poll(self):
                count['n'] += 1
                self.record_commands.focused = count['n'] == 1
                return {'direction': [1., 0., 0.], 'enabled': True}
            def pop_commands(self): return ['leader_toggle'] if count['n'] == 1 else []
            def close(self): pass

        class Reader:
            def latest(self): return sample([0.] * 7, time.monotonic())

        class Viewer:
            user_scn = mujoco.MjvScene(sim.backend.model, maxgeom=50)
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def is_running(self): return count['n'] < 2
            def sync(self): pass

        def tick(start, deadline):
            events.append((sim.leader_active, sim.enabled, sim.leader_status))

        with patch('scripts.native_viewer.NativeInput', Inputs), \
             patch('scripts.native_viewer.prepare_auxiliary_view', return_value=(None, None, None, None)), \
             patch('scripts.native_viewer.viewer_windows', return_value=set()), \
             patch('scripts.native_viewer.mujoco.viewer.launch_passive', return_value=Viewer()), \
             patch.object(sim, 'tick', side_effect=tick):
            show_native_viewer(sim, Reader(), CONFIG)
        self.assertTrue(events[0][0])
        self.assertFalse(events[1][0])
        self.assertFalse(events[1][1])
        self.assertIn('FOCUS LOST', events[1][2])

    def test_viewer_requires_both_manual_deadmen_released_after_leader_stops(self):
        for reason in ('toggle', 'focus', 'stale'):
            with self.subTest(reason=reason):
                sim = Simulation()
                events, count = [], {'n': 0}

                class Inputs:
                    record_commands = SimpleNamespace(focused=True)
                    manual_released = True

                    def poll(self):
                        count['n'] += 1
                        n = count['n']
                        self.record_commands.focused = not (reason == 'focus' and n == 2)
                        # n=3: both held; n=4 and n=5: only one held;
                        # n=6: both released. A disabled sample at n=3 can
                        # hide a held deadman from a different device.
                        self.manual_released = n in (1, 6)
                        return {'direction': [1., 0., 0.], 'enabled': n in (4, 5, 7)}

                    def pop_commands(self):
                        n = count['n']
                        return ['leader_toggle'] if n == 1 or (reason == 'toggle' and n == 2) else []

                    def close(self): pass

                class Reader:
                    def latest(self): return sample([0.] * 7, time.monotonic())

                class Viewer:
                    user_scn = mujoco.MjvScene(sim.backend.model, maxgeom=50)
                    def __enter__(self): return self
                    def __exit__(self, *args): return False
                    def is_running(self): return count['n'] < 7
                    def sync(self): pass

                def tick(start, deadline):
                    events.append(sim.enabled)
                    if reason == 'stale' and count['n'] == 2:
                        sim.stop_leader('STALE - PRESS F11')

                with patch('scripts.native_viewer.NativeInput', Inputs), \
                     patch('scripts.native_viewer.prepare_auxiliary_view', return_value=(None, None, None, None)), \
                     patch('scripts.native_viewer.viewer_windows', return_value=set()), \
                     patch('scripts.native_viewer.mujoco.viewer.launch_passive', return_value=Viewer()), \
                     patch.object(sim, 'tick', side_effect=tick):
                    show_native_viewer(sim, Reader(), CONFIG)
                self.assertEqual(events, [False] * 6 + [True])


if __name__ == '__main__':
    unittest.main()

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from arm_teleop.b601_scene import make_b601_scene_profile
from arm_teleop.mujoco_backend import MujocoBackend
from arm_teleop.recorder import (EpisodeRecorder, RecordingError, _sha256, capture_state,
                                 recording_metadata)
from arm_teleop.server import DT, Simulation


def lines(path):
    return [json.loads(row) for row in path.read_text(encoding='utf-8').splitlines()]


class RecorderTests(unittest.TestCase):
    def test_fingerprint_ignores_windows_line_endings(self):
        with tempfile.TemporaryDirectory() as folder:
            lf, crlf = Path(folder) / 'lf', Path(folder) / 'crlf'
            lf.write_bytes(b'first\nsecond\n')
            crlf.write_bytes(b'first\r\nsecond\r\n')
            self.assertEqual(_sha256(lf), _sha256(crlf))

    def test_transition_is_aligned_and_snapshot_is_copied(self):
        sim = Simulation()
        with tempfile.TemporaryDirectory() as folder:
            recorder = EpisodeRecorder(folder, recording_metadata(sim, 'base', DT))
            state = capture_state(sim)
            sim.backend.step(5, return_state=False)
            next_state = capture_state(sim)
            recorder.append(state, {'direction': [0, 0, 0], 'enabled': False}, next_state)
            sim.backend.data.qpos[:] = 2
            recorder.finish('success', 'marked_by_user')
            meta, transition, end = lines(recorder.path)
            self.assertEqual(meta['schema_version'], 1)
            self.assertEqual(transition['index'], 0)
            self.assertEqual(transition['state']['mj']['time'], 0)
            self.assertAlmostEqual(transition['next_state']['mj']['time'], DT)
            self.assertNotEqual(transition['state']['mj']['qpos'], [2] * sim.backend.model.nq)
            self.assertEqual((end['outcome'], end['reason'], end['steps']),
                             ('success', 'marked_by_user', 1))

    def test_b601_snapshot_includes_free_cube(self):
        root = Path(__file__).resolve().parents[1]
        profile = root / 'robot_models/b601_rs/profile.json'
        if not profile.exists():
            self.skipTest('B601 model not installed')
        sim = Simulation()
        with tempfile.TemporaryDirectory() as folder:
            sim.backend = MujocoBackend('b601_rs', make_b601_scene_profile(profile, folder))
            state = capture_state(sim)
            self.assertEqual(len(state['cube']['position']), 3)
            self.assertEqual(len(state['cube']['quaternion']), 4)
            self.assertAlmostEqual(state['cube']['position'][0], .4)
            self.assertEqual(len(state['mj']['qpos']), sim.backend.model.nq)

    def test_queue_overflow_cannot_produce_success_end(self):
        from arm_teleop import recorder as module
        original = module._line
        entered, release = threading.Event(), threading.Event()
        def slow_line(row):
            if row.get('type') == 'transition':
                entered.set()
                release.wait(3)
            return original(row)
        with tempfile.TemporaryDirectory() as folder, patch.object(module, '_line', slow_line):
            recorder = EpisodeRecorder(folder, {}, queue_size=1)
            recorder.append({}, {}, {})
            self.assertTrue(entered.wait(3))
            recorder.append({}, {}, {})
            with self.assertRaises(RecordingError):
                recorder.append({}, {}, {})
            release.set()
            with self.assertRaises(RecordingError):
                recorder.finish('success', 'user')
            self.assertNotIn('end', [row['type'] for row in lines(recorder.path)])

    def test_serialization_error_is_reported_and_cannot_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = EpisodeRecorder(folder, {})
            recorder.append({}, {'invalid': np.nan}, {})
            with self.assertRaises(RecordingError):
                recorder.finish('success', 'user')
            self.assertNotIn('end', [row['type'] for row in lines(recorder.path)])


if __name__ == '__main__':
    unittest.main()

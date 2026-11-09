"""Exercise actual native commands through the physics recording loop."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco

from arm_teleop.server import Simulation
from scripts.native_viewer import show_native_viewer


class RecordingIntegrationTest(unittest.TestCase):
    def test_recording_failure_does_not_stop_physics(self):
        sim = Simulation()

        class BrokenRecorder:
            def append(self, *args):
                raise OSError('disk unavailable')

        sim.recorder = BrokenRecorder()
        sim.set_input({'direction': [0., 0., 0.], 'enabled': False})
        now = time.monotonic()
        sim.tick(now, now)
        self.assertIsNone(sim.fault)
        self.assertIn('disk unavailable', sim.recording_error)
        first = sim.backend.data.time
        now = time.monotonic()
        sim.tick(now, now)
        self.assertGreater(sim.backend.data.time, first)

    def test_closing_viewer_marks_active_recording_interrupted(self):
        sim = Simulation()
        count = {'poll': 0}

        class Inputs:
            def poll(self):
                count['poll'] += 1
                return {'direction': [0., 0., 0.], 'enabled': False}
            def pop_commands(self): return ['start'] if count['poll'] == 1 else []
            def close(self): pass

        class Viewer:
            user_scn = mujoco.MjvScene(sim.backend.model, maxgeom=50)
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def is_running(self): return count['poll'] < 2
            def sync(self): pass

        with tempfile.TemporaryDirectory() as directory, \
             patch('scripts.native_viewer.RECORDING_DIR', Path(directory)), \
             patch('scripts.native_viewer.NativeInput', Inputs), \
             patch('scripts.native_viewer.prepare_auxiliary_view', return_value=(None, None, None, None)), \
             patch('scripts.native_viewer.viewer_windows', return_value=set()), \
             patch('scripts.native_viewer.mujoco.viewer.launch_passive', return_value=Viewer()):
            show_native_viewer(sim)
            path = next(Path(directory).glob('*.jsonl'))
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(rows[-1]['outcome'], 'interrupted')
        self.assertEqual(rows[-1]['reason'], 'viewer_closed')

    def test_start_record_and_label_in_viewer(self):
        sim = Simulation()
        count = {'poll': 0}

        class Inputs:
            def poll(self):
                count['poll'] += 1
                return {'direction': [1., 0., 0.], 'rotation': [0., 0., 0.], 'enabled': True}
            def pop_commands(self):
                return {1: ['start'], 3: ['success']}.get(count['poll'], [])
            def close(self): pass

        class Viewer:
            user_scn = mujoco.MjvScene(sim.backend.model, maxgeom=50)
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def is_running(self): return count['poll'] < 3
            def sync(self): pass

        with tempfile.TemporaryDirectory() as directory, \
             patch('scripts.native_viewer.RECORDING_DIR', Path(directory)), \
             patch('scripts.native_viewer.NativeInput', Inputs), \
             patch('scripts.native_viewer.prepare_auxiliary_view', return_value=(None, None, None, None)), \
             patch('scripts.native_viewer.viewer_windows', return_value=set()), \
             patch('scripts.native_viewer.mujoco.viewer.launch_passive', return_value=Viewer()):
            show_native_viewer(sim)
            files = list(Path(directory).glob('*.jsonl'))
            self.assertEqual(len(files), 1)
            rows = [json.loads(line) for line in files[0].read_text(encoding='utf-8').splitlines()]
        self.assertEqual(rows[0]['type'], 'metadata')
        self.assertEqual(rows[-1]['outcome'], 'success')
        self.assertEqual(rows[-1]['steps'], 2)
        self.assertEqual(rows[1]['action']['direction'], [1., 0., 0.])
        self.assertTrue(rows[1]['action']['enabled'])
        self.assertAlmostEqual(rows[1]['next_state']['mj']['time'], rows[2]['state']['mj']['time'])
        self.assertGreater(rows[1]['next_state']['mj']['time'], rows[1]['state']['mj']['time'])


if __name__ == '__main__':
    unittest.main()

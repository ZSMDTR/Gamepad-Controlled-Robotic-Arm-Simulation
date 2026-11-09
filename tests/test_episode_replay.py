import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco

from arm_teleop.b601_scene import make_b601_scene_profile
from arm_teleop.mujoco_backend import MujocoBackend
from arm_teleop.recorder import EpisodeRecorder, capture_state, recording_metadata
from arm_teleop.server import DT, Simulation
from scripts.replay_episode import apply_state, inspect_episode, latest_episode, make_backend


class EpisodeReplayTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'episode.jsonl'
        self.model = mujoco.MjModel.from_xml_string('''
            <mujoco><worldbody><body><freejoint/>
            <geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>''')
        self.data = mujoco.MjData(self.model)

    def state(self, moment, x=0):
        self.data.qpos[0] = x
        return {'mj': {'time': moment,
                       'qpos': self.data.qpos.tolist(), 'qvel': self.data.qvel.tolist(),
                       'act': self.data.act.tolist(), 'ctrl': self.data.ctrl.tolist(),
                       'mocap_pos': self.data.mocap_pos.tolist(),
                       'mocap_quat': self.data.mocap_quat.tolist()}}

    def write(self, rows):
        self.path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    def rows(self):
        return [{'type': 'metadata', 'schema_version': 1, 'robot_id': 'b601_rs', 'scene_id': 'b601_practice'},
                {'type': 'transition', 'index': 0, 'state': self.state(0),
                 'action': {'direction': [0, 0, 0], 'rotation': [0, 0, 0],
                            'gripper': 0, 'enabled': False},
                 'next_state': self.state(.01, .1)},
                {'type': 'end', 'outcome': 'success', 'steps': 1}]

    def test_inspect_and_restore_recorded_object_position(self):
        rows = self.rows()
        self.write(rows)
        metadata, end = inspect_episode(self.path)
        self.assertEqual(metadata['scene_id'], 'b601_practice')
        self.assertEqual(end['outcome'], 'success')
        apply_state(self.model, self.data, rows[1]['next_state'])
        self.assertAlmostEqual(self.data.time, .01)
        self.assertAlmostEqual(self.data.qpos[0], .1)
        self.assertAlmostEqual(self.data.xpos[1, 0], .1)

    def test_cli_inspect_does_not_require_graphics_or_installed_robot(self):
        self.write(self.rows())
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root / 'scripts/replay_episode.py'),
                                 str(self.path), '--inspect'], cwd=root,
                                text=True, encoding='utf-8', capture_output=True,
                                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('b601_practice', result.stdout)
        self.assertIn('状态回放不等于动作重演', result.stdout)

    def test_truncated_or_wrong_count_is_rejected(self):
        rows = self.rows()
        self.write(rows[:-1])
        with self.assertRaisesRegex(ValueError, '没有结束标记'):
            inspect_episode(self.path)
        rows[-1]['steps'] = 3
        self.write(rows)
        with self.assertRaisesRegex(ValueError, '步数'):
            inspect_episode(self.path)

    def test_latest_uses_newest_and_explains_empty_directory(self):
        with self.assertRaisesRegex(ValueError, '没有找到示教轨迹'):
            latest_episode(self.tmp.name)
        first = Path(self.tmp.name) / 'episode_first.jsonl'
        second = Path(self.tmp.name) / 'episode_second.jsonl'
        first.touch()
        second.touch()
        os.utime(first, (1_600_000_000, 1_600_000_000))
        os.utime(second, (1_600_000_001, 1_600_000_001))
        self.assertEqual(latest_episode(self.tmp.name), second)
    def test_mismatched_model_or_nonfinite_state_is_rejected(self):
        state = self.state(0)
        state['mj']['qpos'] = [0]
        with self.assertRaisesRegex(ValueError, '尺寸与当前模型不一致'):
            apply_state(self.model, self.data, state)
        state['mj']['qpos'] = [float('nan')] * self.model.nq
        with self.assertRaisesRegex(ValueError, '非有限数值'):
            apply_state(self.model, self.data, state)

    def test_model_fingerprint_mismatch_is_rejected_before_playback(self):
        meta = {'robot_id': 'b601_rs', 'scene_id': 'b601_practice',
                'model_fingerprints': {'profile_sha256': 'old', 'model_xml_sha256': 'same'}}
        with patch('scripts.replay_episode.model_fingerprints',
                   return_value={'profile_sha256': 'new', 'model_xml_sha256': 'same'}):
            with self.assertRaisesRegex(ValueError, '模型.*不同'):
                make_backend(meta, self.tmp.name)

    def test_backward_time_and_extra_data_are_rejected(self):
        rows = self.rows()
        rows[1]['next_state']['mj']['time'] = -1
        self.write(rows)
        with self.assertRaisesRegex(ValueError, '时间倒退'):
            inspect_episode(self.path)
        rows = self.rows() + [{'type': 'transition', 'index': 1}]
        self.write(rows)
        with self.assertRaisesRegex(ValueError, '结束标记之后'):
            inspect_episode(self.path)

    def test_invalid_actions_are_rejected(self):
        for field, bad in [('direction', [0, 0]), ('rotation', [0, float('nan'), 0]),
                           ('gripper', 2), ('enabled', 1)]:
            rows = self.rows()
            rows[1]['action'][field] = bad
            self.write(rows)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                inspect_episode(self.path)

    def test_real_b601_recording_roundtrip_with_local_scene(self):
        profile = Path(__file__).resolve().parents[1] / 'robot_models/b601_rs/profile.json'
        if not profile.is_file():
            self.skipTest('B601 model not installed')
        sim = Simulation()
        sim.backend = MujocoBackend('b601_rs', make_b601_scene_profile(profile, self.tmp.name))
        recording = EpisodeRecorder(self.tmp.name, recording_metadata(sim, 'b601_practice', DT))
        initial = capture_state(sim)
        sim.backend.step(5, return_state=False)
        final = capture_state(sim)
        recording.append(initial, {'enabled': False, 'direction': [0, 0, 0],
                                   'rotation': [0, 0, 0], 'gripper': 0}, final)
        recording.finish('interrupted', 'test')
        metadata, _ = inspect_episode(recording.path)
        backend = make_backend(metadata, self.tmp.name)
        transition = json.loads(recording.path.read_text(encoding='utf-8').splitlines()[1])
        apply_state(backend.model, backend.data, transition['next_state'])
        self.assertAlmostEqual(backend.data.time, final['mj']['time'])
        self.assertAlmostEqual(backend.data.qpos[-7], final['mj']['qpos'][-7])


if __name__ == '__main__':
    unittest.main()

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from arm_teleop.leader_home import compare_home, load_home, save_home


def sample(positions, timestamp):
    return SimpleNamespace(positions=positions, timestamp=timestamp)


def stable(positions=None, now=10.):
    positions = positions or [0.] * 7
    return [sample(positions, now - offset) for offset in (.36, .27, .18, .09, 0.)]


class LeaderHomeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'leader-home.json'

    def test_save_load_compare_and_each_axis_tolerance(self):
        positions = [.1, -.2, .3, -.4, .5, -.6, .7]
        home = save_home(self.path, stable(positions), 10.)
        self.assertEqual(load_home(self.path), home)
        for axis in range(7):
            inside = positions.copy()
            inside[axis] += math.radians(4.9)
            self.assertTrue(compare_home(home, sample(inside, 10.), 10.)[0])
            outside = positions.copy()
            outside[axis] += math.radians(5.1)
            ok, errors = compare_home(home, sample(outside, 10.), 10.)
            self.assertFalse(ok)
            self.assertAlmostEqual(errors[axis], math.radians(5.1))

    def test_wrap_around_plus_minus_pi(self):
        home = save_home(self.path, stable([math.pi - .01] * 7), 10.)
        ok, errors = compare_home(home, sample([-math.pi + .01] * 7, 10.), 10.)
        self.assertTrue(ok)
        self.assertTrue(all(abs(error - .02) < 1e-10 for error in errors))

    def test_missing_corrupt_and_invalid_file_fail_closed(self):
        with self.assertRaises(FileNotFoundError):
            load_home(self.path)
        for text in ('{', '{}', json.dumps({'version': 1, 'positions_rad': [0] * 7,
                                             'tolerances_rad': [0] * 7}),
                     json.dumps({'version': True, 'positions_rad': [0] * 7,
                                 'tolerances_rad': [.1] * 7})):
            self.path.write_text(text, encoding='utf-8')
            with self.assertRaises(ValueError):
                load_home(self.path)

    def test_failed_save_preserves_existing_calibration(self):
        home = save_home(self.path, stable(), 10.)
        moving = stable()
        moving[0] = sample([math.radians(2)] + [0.] * 6, moving[0].timestamp)
        bad_sets = [[], stable(now=9.), stable()[-2:], moving,
                    [sample([0.] * 7, 9.6), sample([0.] * 7, 10.)],
                    [sample([0.] * 7, 9.7), sample([0.] * 7, 9.7), sample([0.] * 7, 10.)],
                    stable([float('nan')] * 7), stable([math.pi + .01] * 7)]
        for samples in bad_sets:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                save_home(self.path, samples, 10.)
            self.assertEqual(load_home(self.path), home)

    def test_stale_missing_and_malformed_samples_fail_closed(self):
        home = save_home(self.path, stable(), 10.)
        for bad in (None, sample([0.] * 7, 9.), sample([0.] * 6, 10.),
                    sample([0.] * 7, 10.1), sample([True] * 7, 10.),
                    sample([float('inf')] * 7, 10.)):
            with self.subTest(sample=bad), self.assertRaises(ValueError):
                compare_home(home, bad, 10.)


if __name__ == '__main__':
    unittest.main()

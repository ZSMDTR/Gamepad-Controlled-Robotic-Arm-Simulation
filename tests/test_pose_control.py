import unittest
import numpy as np

from arm_teleop.pose_control import PoseController, rotation_exp, rotation_log, sdls_pose, _box_dls
from arm_teleop.solvers import sdls


def controller(n=6, **settings):
    spec = dict(lower=[-2.] * n, upper=[2.] * n, velocity_limit=[1.] * n,
                acceleration_limit=[4.] * n, home=[0.] * n)
    spec.update(settings)
    return PoseController(spec)


def tick(c, q=None, tcp=None, rot=None, jac=None, move=None, turn=None, enabled=True, dt=.01):
    n = c.n
    if jac is None:
        jac = np.zeros((6, n))
        jac[:, :min(6, n)] = np.eye(6)[:, :min(6, n)]
    return c.step(np.zeros(n) if q is None else q, np.zeros(n),
                  np.zeros(3) if tcp is None else tcp, np.eye(3) if rot is None else rot,
                  jac, np.zeros(3) if move is None else move,
                  np.zeros(3) if turn is None else turn, enabled, dt)


class PoseControlTests(unittest.TestCase):
    def test_bounded_dls_improves_task_error_over_joint_clipping(self):
        # A saturated joint can be compensated by the other joint. Ordinary
        # DLS followed by independent clipping cannot make that correction.
        j = np.array([[1., 1.], [0., 1.]])
        target = np.array([1., .5])
        damping = .04
        lower, upper = np.array([-2., -2.]), np.array([.2, 2.])
        baseline = np.clip(np.linalg.solve(j.T @ j + damping**2 * np.eye(2), j.T @ target), lower, upper)
        result, converged = _box_dls(j, target, damping, lower, upper, np.zeros(2))
        self.assertTrue(converged)
        self.assertLess(np.linalg.norm(j @ result - target), np.linalg.norm(j @ baseline - target) * .8)
        self.assertTrue(np.all(result >= lower) and np.all(result <= upper))

    def test_box_dls_satisfies_kkt_for_random_singular_and_redundant_tasks(self):
        rng = np.random.default_rng(928)
        for rows, cols in ((3, 7), (6, 6), (6, 2)):
            for _ in range(30):
                j = rng.normal(size=(rows, cols))
                j[:, 0] = 0.  # Include a rank-deficient direction.
                target = rng.normal(size=rows)
                lower = rng.uniform(-.7, 0, size=cols)
                upper = rng.uniform(0, .7, size=cols)
                center = rng.normal(size=cols)
                x, converged = _box_dls(j, target, .04, lower, upper, center)
                self.assertTrue(converged)
                self.assertTrue(np.all(x >= lower - 1e-10) and np.all(x <= upper + 1e-10))
                gradient = j.T @ (j @ x - target) + .04**2 * (x - center)
                for i in range(cols):
                    if x[i] <= lower[i] + 1e-8:
                        self.assertGreaterEqual(gradient[i], -1e-8)
                    elif x[i] >= upper[i] - 1e-8:
                        self.assertLessEqual(gradient[i], 1e-8)
                    else:
                        self.assertAlmostEqual(gradient[i], 0., delta=1e-8)

    def test_constrained_mode_respects_velocity_acceleration_and_reference(self):
        c = controller(mode='constrained_dls', velocity_limit=[.03] * 6,
                       acceleration_limit=[.2] * 6, max_joint_reference_lead=.02, posture_gain=0)
        previous = np.zeros(6)
        for _ in range(100):
            state = tick(c, move=[1, 1, 0], turn=[0, 0, 1])
            self.assertNotIn('infeasible_velocity_bounds', state['constraints'])
            self.assertNotIn('solver_iteration_limit', state['constraints'])
            self.assertTrue(np.all(np.abs(state['qdot_cmd']) <= .03 + 1e-10))
            self.assertTrue(np.all(np.abs(state['qdot_cmd'] - previous) <= .002 + 1e-10))
            self.assertTrue(np.all(np.abs(state['q_ref']) <= .02 + 1e-10))
            previous = state['qdot_cmd'].copy()

    def test_conflicting_bounds_fall_back_and_report_conflict(self):
        c = controller(mode='constrained_dls', acceleration_limit=[.2] * 6, posture_gain=0)
        c.q_ref = np.zeros(6)
        c.velocity[:] = .1
        # q moved abruptly beyond the reference-lead range. Acceleration and
        # reference bounds cannot both be met in a single cycle.
        previous_ref = c.q_ref.copy()
        state = tick(c, q=np.ones(6), move=[1, 0, 0])
        self.assertIn('infeasible_velocity_bounds', state['constraints'])
        self.assertIn('reference_lead_and_braking_relaxed', state['constraints'])
        np.testing.assert_allclose((state['q_ref'] - previous_ref) / .01, state['qdot_cmd'])
        self.assertTrue(np.all(np.abs(state['qdot_cmd']) <= c.vmax + 1e-10))
        self.assertTrue(np.all(np.abs(state['qdot_cmd'] - .1) <= .002 + 1e-10))
        self.assertTrue(np.isfinite(state['q_ref']).all())

    def test_deadman_with_measured_state_jump_keeps_reference_continuous(self):
        c = controller(n=1, mode='constrained_dls', acceleration_limit=[.2], posture_gain=0)
        tick(c)
        previous_ref = c.q_ref.copy()
        c.velocity[:] = .1
        result = tick(c, q=[-.5], enabled=False)
        self.assertIn('hold', result['constraints'])
        self.assertIn('deadman_reference_lead_conflict', result['constraints'])
        self.assertIn('reference_lead_and_braking_relaxed', result['constraints'])
        np.testing.assert_allclose((result['q_ref'] - previous_ref) / .01, result['qdot_cmd'])
        self.assertLessEqual(abs(result['qdot_cmd'][0] - .1), .002 + 1e-10)
        self.assertLessEqual(abs(result['q_ref'][0] - previous_ref[0]), .01 + 1e-10)

    def test_conflict_at_hard_joint_bound_reports_acceleration_relaxation(self):
        c = controller(n=1, mode='constrained_dls', acceleration_limit=[.2], posture_gain=0)
        c.q_ref = np.array([2.])
        c.velocity[:] = .1
        state = tick(c, q=[-1.])
        self.assertIn('infeasible_velocity_bounds', state['constraints'])
        self.assertIn('acceleration_relaxed_at_joint_limit', state['constraints'])
        np.testing.assert_allclose(state['q_ref'], [2.])
        np.testing.assert_allclose(state['qdot_cmd'], [0.])

    def test_exp_log_for_axes_and_near_pi(self):
        for axis in np.eye(3):
            for angle in (0, 1e-8, .4, np.pi - 1e-7, np.pi):
                r = rotation_exp(axis * angle)
                self.assertTrue(np.allclose(rotation_exp(rotation_log(r)), r, atol=2e-7))
        with self.assertRaises(ValueError):
            rotation_log(np.diag([1., 1., -1.]))

    def test_each_translation_and_rotation_axis(self):
        for mode in ('dls', 'sdls', 'position_priority'):
            for i in range(6):
                c = controller(mode=mode, posture_gain=0)
                move, turn = np.zeros(3), np.zeros(3)
                (move if i < 3 else turn)[i % 3] = 1
                state = tick(c, move=move, turn=turn)
                self.assertGreater(state['qdot_cmd'][i], 0)
                self.assertTrue(np.allclose(np.delete(state['qdot_cmd'], i), 0))

    def test_enabled_pose_target_persists_when_sticks_released(self):
        c = controller(posture_gain=0)
        moving = tick(c, move=[1, 0, 0], turn=[0, 0, 1])
        held = tick(c)
        np.testing.assert_allclose(held['target_position'], moving['target_position'])
        np.testing.assert_allclose(held['target_rotation'], moving['target_rotation'])
        self.assertGreater(held['error_m'], 0)
        self.assertGreater(held['error_rad'], 0)

    def test_deadman_decelerates_and_reanchors(self):
        c = controller(posture_gain=0, acceleration_limit=[.5] * 6)
        for _ in range(10):
            moving = tick(c, move=[1, 0, 0])
        paused = tick(c, enabled=False)
        self.assertGreater(paused['qdot_cmd'][0], 0)
        self.assertLessEqual(abs(paused['qdot_cmd'][0] - moving['qdot_cmd'][0]), .005 + 1e-12)
        np.testing.assert_allclose(paused['target_position'], 0)
        resumed = tick(c)
        np.testing.assert_allclose(resumed['target_position'], 0)

    def test_reference_and_target_lead_bounds(self):
        c = controller(max_position_lead=.01, max_orientation_lead=.05, max_joint_reference_lead=.02, posture_gain=0)
        for _ in range(300):
            state = tick(c, move=[1, 0, 0], turn=[0, 0, 1])
        self.assertLessEqual(state['error_m'], .0100000001)
        self.assertLessEqual(state['error_rad'], .0500000001)
        self.assertLessEqual(np.max(np.abs(state['q_ref'])), .020000001)

    def test_model_dimension_acceleration_and_velocity_limits(self):
        c = controller(n=7, velocity_limit=[.03] * 7, acceleration_limit=[.2] * 7, posture_gain=0)
        previous = np.zeros(7)
        for _ in range(30):
            state = tick(c, move=[1, 1, 1], turn=[1, 1, 1])
            v = state['qdot_cmd']
            self.assertEqual(v.shape, (7,))
            self.assertLessEqual(np.max(np.abs(v)), .03000001)
            self.assertLessEqual(np.max(np.abs(v - previous)), .00200001)
            previous = v.copy()

    def test_position_priority_projects_rotation(self):
        c = controller(mode='position_priority', posture_gain=0)
        # One joint moves both position and orientation; another angular-only.
        j = np.zeros((6, 6))
        j[0, 0], j[3, 0], j[3, 1] = 1, 1, 1
        result = tick(c, jac=j, turn=[1, 0, 0])
        self.assertAlmostEqual(result['qdot_cmd'][0], 0, places=10)
        self.assertGreater(result['qdot_cmd'][1], 0)

    def test_sdls_single_block_matches_original_implementation(self):
        rng = np.random.default_rng(1949)
        for _ in range(25):
            j = rng.normal(size=(3, 7))
            e = rng.normal(size=3) * .1
            np.testing.assert_allclose(sdls_pose(j, e), sdls(j, e), atol=1e-12)

    def test_sdls_singular_pose_stays_finite_and_bounded(self):
        j = np.zeros((6, 7))
        j[0, 0] = 1.
        j[4, 1] = 1e-12
        step = sdls_pose(j, np.ones(6), gamma_max=.1)
        self.assertTrue(np.isfinite(step).all())
        self.assertLessEqual(np.max(np.abs(step)), .100000001)
        self.assertEqual(step[1], 0.)
        np.testing.assert_array_equal(sdls_pose(np.zeros((6, 7)), np.ones(6)), np.zeros(7))

    def test_invalid_input_rejected(self):
        c = controller()
        with self.assertRaises(ValueError):
            tick(c, move=[np.nan, 0, 0])
        with self.assertRaises(ValueError):
            tick(c, move=[2, 0, 0])
        with self.assertRaises(ValueError):
            tick(c, dt=0)
        with self.assertRaises(ValueError):
            tick(c, jac=np.full((6, 6), np.nan))


if __name__ == '__main__':
    unittest.main()

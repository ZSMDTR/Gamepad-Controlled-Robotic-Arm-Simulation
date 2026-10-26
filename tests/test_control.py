import json
import unittest

import numpy as np

from arm_teleop.kinematics import ArmModel
from arm_teleop.controller import Controller
from arm_teleop.solvers import dls, sdls, nullspace_projector


class KinematicsTests(unittest.TestCase):
    def test_jacobian_matches_finite_difference(self):
        model = ArmModel()
        rng = np.random.default_rng(41)
        for q in [model.home, *rng.uniform(-.8, .8, size=(8, 6))]:
            numeric = np.empty((3, 6))
            for k in range(6):
                shift = np.zeros(6)
                shift[k] = 1e-6
                numeric[:, k] = (model.fk(q + shift)[-1] - model.fk(q - shift)[-1]) / 2e-6
            np.testing.assert_allclose(model.jacobian(q), numeric, atol=2e-9)

    def test_local_link_lengths_are_preserved(self):
        model = ArmModel()
        np.testing.assert_allclose(np.linalg.norm(np.diff(model.fk(model.home), axis=0), axis=1),
                                   np.linalg.norm(model.offsets, axis=1), atol=1e-12)


class SolverTests(unittest.TestCase):
    def test_dls_hand_calculation(self):
        actual = dls(np.diag([1., .02]), np.array([.01, .01]), .1)
        np.testing.assert_allclose(actual, [.009900990099, .019230769231], atol=1e-12)

    def test_sdls_diagonal_clamps_each_mode(self):
        actual = sdls(np.diag([1., .02, 0.]), [.01, .01, 1.], gamma_max=.1)
        np.testing.assert_allclose(actual, [.01, .1, 0], atol=1e-12)

    def test_sdls_singular_is_finite_and_bounded(self):
        for j in [np.zeros((3, 6)), ArmModel().jacobian(np.zeros(6)),
                  np.array([[1, 1, 0], [0, 1e-11, 0], [0, 0, 0]])]:
            answer = sdls(j, [1, -2, 1], gamma_max=.07)
            self.assertTrue(np.isfinite(answer).all())
            self.assertLessEqual(np.max(np.abs(answer)), .07 + 1e-12)

    def test_posture_projection_preserves_first_order_position(self):
        model = ArmModel()
        j = model.jacobian(model.home)
        n = nullspace_projector(j)
        np.testing.assert_allclose(j @ n, 0, atol=1e-12)
        np.testing.assert_allclose(n @ n, n, atol=1e-12)
        self.assertAlmostEqual(np.trace(n), 3)


class ControllerTests(unittest.TestCase):
    def test_repeated_input_respects_bounds_for_both_methods(self):
        for method in ("dls", "sdls"):
            ctrl = Controller(method)
            previous_velocity = np.zeros(6)
            for k in range(900):
                # Includes sustained unreachable outward intent and reversals.
                direction = [1., 0, 0] if k < 500 else [-1., .7, -.4]
                state = ctrl.step(direction, True, .01)
                velocity = np.array(state["joint_velocity"])
                self.assertTrue(np.isfinite(ctrl.q).all())
                self.assertTrue(np.all(ctrl.q >= ctrl.model.lower))
                self.assertTrue(np.all(ctrl.q <= ctrl.model.upper))
                self.assertLessEqual(state["error_m"], ctrl.max_target_lead + 1e-12)
                self.assertTrue(np.all(np.abs(velocity) <= ctrl.model.velocity_limit + 1e-10))
                self.assertTrue(np.all(np.abs(velocity - previous_velocity) <= ctrl.model.acceleration_limit * .01 + 1e-10))
                previous_velocity = velocity

    def test_neutral_and_deadman_freeze_without_reference_windup(self):
        for enabled, direction in [(False, [1, 0, 0]), (True, [0, 0, 0])]:
            ctrl = Controller()
            for _ in range(40):
                ctrl.step([0, 1, .2], True, .01)
            q = ctrl.q.copy()
            for _ in range(30):
                state = ctrl.step(direction, enabled, .01)
                np.testing.assert_array_equal(ctrl.q, q)
                np.testing.assert_array_equal(state["tcp"], state["target"])
                np.testing.assert_array_equal(ctrl.qdot, 0)

    def test_invalid_commands_do_not_mutate_state(self):
        ctrl = Controller()
        before = json.dumps(ctrl.state(), sort_keys=True)
        for direction, dt in [([float("nan"), 0, 0], .01), ([2, 0, 0], .01),
                              ([1, 0], .01), ([1, 0, 0], 0), ([1, 0, 0], .1)]:
            with self.assertRaises(ValueError):
                ctrl.step(direction, True, dt)
            self.assertEqual(before, json.dumps(ctrl.state(), sort_keys=True))


if __name__ == "__main__":
    unittest.main()

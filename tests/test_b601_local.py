"""Local-only B601-RS simulation checks (requires separately supplied model assets)."""
import unittest
from pathlib import Path

import mujoco
import numpy as np

from arm_teleop.mujoco_backend import MujocoBackend
from arm_teleop.pose_control import PoseController, rotation_log


MODEL = Path(__file__).resolve().parents[1] / "robot_models" / "b601_rs" / "profile.json"


@unittest.skipUnless(MODEL.is_file(), "B601-RS local model assets not installed")
class B601LocalTests(unittest.TestCase):
    def test_home_is_stable_and_starts_without_contacts(self):
        b = MujocoBackend("b601_rs")
        self.assertEqual(b.data.ncon, 0)
        initial = b.state()
        b.command(initial["q"], initial["gripper_position"])
        b.step(1000)
        state = b.state()
        self.assertLess(np.max(np.abs(state["q"] - initial["q"])), 1e-3)
        self.assertLess(np.max(np.abs(state["qvel"])), 1e-3)
        self.assertFalse(any(w.number for w in b.data.warning))

    def test_gripper_closes_and_opens(self):
        b = MujocoBackend("b601_rs")
        b.command(b.spec["home"], 0.)
        b.step(2000)
        self.assertLess(b.state()["gripper_position"], .05)
        b.command(b.spec["home"], 1.)
        b.step(2500)
        self.assertGreater(b.state()["gripper_position"], .95)
        self.assertTrue(np.isfinite(b.data.qpos).all())

    def test_translation_jacobian_matches_finite_difference(self):
        b = MujocoBackend("b601_rs")
        initial = b.state()
        for index, address in enumerate(b.qpos_ids):
            original = b.data.qpos[address]
            b.data.qpos[address] = original + 1e-6
            mujoco.mj_forward(b.model, b.data)
            plus = b.data.site_xpos[b.tcp_id].copy()
            b.data.qpos[address] = original - 1e-6
            mujoco.mj_forward(b.model, b.data)
            minus = b.data.site_xpos[b.tcp_id].copy()
            b.data.qpos[address] = original
            mujoco.mj_forward(b.model, b.data)
            np.testing.assert_allclose((plus - minus) / 2e-6,
                                       initial["jacobian"][:3, index], atol=1e-6)

    def test_translation_and_rotation_in_both_solver_modes(self):
        for mode in ("dls", "constrained_dls"):
            with self.subTest(mode=mode):
                b = MujocoBackend("b601_rs")
                controller = PoseController(dict(b.spec, mode=mode))
                initial = b.state()
                for i in range(200):
                    state = b.state()
                    translation = [.35, 0, 0] if i < 100 else [0, 0, 0]
                    rotation = [0, 0, 0] if i < 100 else [0, 0, .35]
                    result = controller.step(state["q"], state["qvel"], state["tcp"],
                                             state["rotation"], state["jacobian"],
                                             translation, rotation, True, .01)
                    b.command(result["q_ref"], .5)
                    b.step(5, return_state=False)
                    if i == 99:
                        translated = b.state()
                final = b.state()
                self.assertGreater(translated["tcp"][0] - initial["tcp"][0], .01)
                self.assertGreater(rotation_log(final["rotation"] @ translated["rotation"].T)[2], .05)
                self.assertTrue(np.isfinite(final["q"]).all())
                self.assertFalse(any(w.number for w in b.data.warning))


if __name__ == "__main__":
    unittest.main()

import unittest

import mujoco
import numpy as np

from arm_teleop.world_axes import add_world_axes, update_tcp_axes


class WorldAxesTest(unittest.TestCase):
    def test_labels_and_tcp_guide_stay_world_aligned(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
        scene = mujoco.MjvScene(model, maxgeom=20)
        first = add_world_axes(scene)
        self.assertEqual(scene.ngeom, 9)
        self.assertEqual([scene.geoms[first + 3 + i].label for i in range(3)],
                         ['+X FRONT', '+Y LEFT', '+Z UP'])
        for position in ([.1, .2, .3], [-.2, .3, .4]):
            update_tcp_axes(scene, first, position)
            for axis in range(3):
                np.testing.assert_allclose(scene.geoms[first + axis].mat[:, 2], np.eye(3)[axis])
                arrow = scene.geoms[first + 6 + axis]
                np.testing.assert_allclose(arrow.pos, position, atol=1e-7)
                np.testing.assert_allclose(arrow.mat[:, 2], np.eye(3)[axis])


if __name__ == '__main__':
    unittest.main()

import copy
import json
import socket
import unittest
from urllib.request import urlopen
from unittest.mock import patch

import mujoco
import numpy as np

from arm_teleop.server import Simulation
from scripts.native_viewer import copy_view_data, main, show_viewer


class NativeViewerTest(unittest.TestCase):
    def test_entry_starts_selected_robot_and_opens_browser(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        url = f'http://127.0.0.1:{port}/'

        def check_view(sim):
            with urlopen(url + 'api/state', timeout=2) as response:
                state = json.load(response)
            self.assertEqual(state['robot_id'], 'ur5e')
            self.assertEqual(sim.backend.name, 'ur5e')

        with patch('sys.argv', ['native_viewer.py', '--robot', 'ur5e', '--port', str(port), '--open-browser']), \
             patch('scripts.native_viewer.webbrowser.open') as browser, \
             patch('scripts.native_viewer.show_viewer', check_view):
            main()
        browser.assert_called_once_with(url)

    def test_view_copy_is_independent_and_rejects_replaced_model(self):
        sim = Simulation()
        model = sim.backend.model
        view_model = copy.copy(model)
        view_data = mujoco.MjData(view_model)
        with sim.lock:
            sim.backend.data.qpos[sim.backend.qpos_ids[0]] += .1
            mujoco.mj_forward(model, sim.backend.data)
        self.assertTrue(copy_view_data(sim, model, view_data))
        np.testing.assert_allclose(view_data.qpos, sim.backend.data.qpos)
        view_data.qpos[0] += 1
        self.assertNotEqual(view_data.qpos[0], sim.backend.data.qpos[0])
        sim.configure({'robot': 'panda'})
        self.assertFalse(copy_view_data(sim, model, view_data))

    def test_switching_robots_reopens_and_closing_exits(self):
        sim = Simulation()
        launches = []

        class Viewer:
            def __init__(self, model, data):
                self.model, self.data, self.synced = model, data, False
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def is_running(self): return not self.synced or len(launches) == 1
            def lock(self):
                from contextlib import nullcontext
                return nullcontext()
            def sync(self):
                self.synced = True
                if len(launches) == 1:
                    sim.configure({'robot': 'panda'})

        def launch(model, data, **kwargs):
            launches.append(model)
            return Viewer(model, data)

        with patch('scripts.native_viewer.mujoco.viewer.launch_passive', launch), \
             patch('scripts.native_viewer.time.sleep'):
            show_viewer(sim)
        self.assertEqual(len(launches), 2)
        self.assertEqual((launches[0].nq, launches[1].nq), (8, 9))


if __name__ == '__main__':
    unittest.main()

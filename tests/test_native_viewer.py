import copy
import json
import socket
import time
import unittest
from pathlib import Path
from urllib.request import urlopen
from unittest.mock import patch

import mujoco
import numpy as np

from arm_teleop.server import Simulation
from scripts.native_viewer import copy_view_data, main, show_viewer, show_native_viewer


class NativeViewerTest(unittest.TestCase):
    def test_native_deadman_release_discards_servo_lead(self):
        profile = Path(__file__).resolve().parents[1] / 'robot_models/b601_rs/profile.json'
        if not profile.is_file():
            self.skipTest('B601 model not installed')
        sim = Simulation()
        sim.configure({'robot': 'b601_rs'})
        sim.native_hold_on_release = True
        sim.start()
        try:
            deadline = time.monotonic() + .6
            while time.monotonic() < deadline:
                sim.set_input({'direction': [1., 0., 0.], 'enabled': True})
                time.sleep(.01)
            with sim.lock:
                before = sim.backend.state()['tcp']
            sim.set_input({'direction': [0., 0., 0.], 'enabled': False})
            time.sleep(.7)
            with sim.lock:
                after = sim.backend.state()['tcp']
                self.assertIsNone(sim.fault)
            self.assertLess(np.linalg.norm(after - before), .01)
        finally:
            sim.close()

    def test_b601_default_entry_builds_practice_scene_without_web_server(self):
        profile = Path(__file__).resolve().parents[1] / 'robot_models/b601_rs/profile.json'
        if not profile.is_file():
            self.skipTest('B601 model not installed')

        def check_scene(sim):
            self.assertEqual(sim.backend.name, 'b601_rs')
            self.assertGreaterEqual(mujoco.mj_name2id(sim.backend.model, mujoco.mjtObj.mjOBJ_BODY, 'practice_cube'), 0)

        with patch('sys.argv', ['native_viewer.py', '--robot', 'b601_rs']), \
             patch('scripts.native_viewer.Server', side_effect=AssertionError('unexpected web server')), \
             patch('scripts.native_viewer.show_native_viewer', check_scene):
            main()

    def test_native_viewer_controls_live_simulation_without_server(self):
        sim = Simulation()
        original_model, original_data = sim.backend.model, sim.backend.data
        samples = []

        class Inputs:
            def poll(self):
                samples.append(True)
                return {'direction': [0., 0., 0.], 'enabled': False}
            def close(self):
                samples.append('closed')

        class Viewer:
            user_scn = mujoco.MjvScene(original_model, maxgeom=50)
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def is_running(self): return not samples
            def lock(self):
                raise AssertionError('native viewer must not retain its native lock')
            def sync(self):
                original_data.qpos[0] = .123

        def launch(model, data, **kwargs):
            self.assertIs(model, original_model)
            self.assertIs(data, original_data)
            self.assertTrue(kwargs['show_left_ui'])
            self.assertTrue(kwargs['show_right_ui'])
            return Viewer()

        with patch('scripts.native_viewer.NativeInput', Inputs), \
             patch('scripts.native_viewer.mujoco.viewer.launch_passive', launch), \
             patch('scripts.native_viewer.time.sleep'), \
             patch.object(sim, 'start', side_effect=AssertionError('unexpected physics thread')), \
             patch.object(sim, 'tick', wraps=sim.tick) as tick:
            show_native_viewer(sim)
        tick.assert_called_once()
        self.assertEqual(samples, [True, 'closed'])
        self.assertFalse(sim.thread.is_alive())
        self.assertAlmostEqual(original_data.qpos[0], .123, places=2)

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

import http.client
import json
import threading
import time
import unittest

from arm_teleop.server import Simulation, Server, INPUT_TIMEOUT


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = Simulation()
        cls.server = Server(("127.0.0.1", 0), cls.sim)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.sim.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.sim.close()
        cls.thread.join(timeout=2)

    def request(self, path, body=None, origin=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        connection.request("POST" if body is not None else "GET", path, body, headers)
        response = connection.getresponse()
        status, data = response.status, response.read()
        connection.close()
        return status, json.loads(data) if path.startswith("/api/") else data

    def test_input_expires_without_heartbeat(self):
        self.request("/api/reset", "{}")
        self.assertEqual(self.request("/api/input", '{"direction":[0,1,0],"enabled":true}')[0], 200)
        self.assertTrue(self.request("/api/state")[1]["enabled"])
        time.sleep(INPUT_TIMEOUT + .06)
        state = self.request("/api/state")[1]
        self.assertTrue(state["input_stale"])
        self.assertFalse(state["enabled"])
        self.assertEqual(state["joint_velocity"], [0.] * 6)

    def test_invalid_and_cross_origin_input_rejected(self):
        self.assertEqual(self.request("/api/input", '{"direction":[NaN,0,0],"enabled":true}')[0], 400)
        self.assertEqual(self.request("/api/input", '{"direction":[0,0,0],"enabled":"yes"}')[0], 400)
        self.assertEqual(self.request("/api/config", '{"elbow_weight":2}')[0], 400)
        self.assertEqual(self.request("/api/reset", '{}', 'https://example.com')[0], 403)

    def test_reset_and_configure(self):
        self.assertEqual(self.request("/api/config", '{"method":"dls","elbow_weight":0}')[0], 200)
        self.assertEqual(self.request("/api/state")[1]["method"], "dls")
        self.assertEqual(self.request("/api/reset", "{}")[0], 200)
        state = self.request("/api/state")[1]
        self.assertEqual(state["tcp"], state["target"])
        self.assertIsNone(state["fault"])

    def test_static_routes_and_no_path_traversal(self):
        for path in ("/", "/app.js", "/styles.css"):
            status, data = self.request(path)
            self.assertEqual(status, 200)
            self.assertGreater(len(data), 100)
        self.assertEqual(self.request("/../pyproject.toml")[0], 404)


if __name__ == "__main__":
    unittest.main()

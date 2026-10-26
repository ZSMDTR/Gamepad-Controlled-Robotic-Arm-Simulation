"""Loopback HTTP UI and a paced, best-effort 100 Hz kinematic simulator.

The browser only supplies intent. Network cadence does not set the solver dt.
This module has no robot driver or hardware command path.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from .controller import Controller

DT = 0.01
INPUT_TIMEOUT = 0.25
STATIC = Path(__file__).parent / "static"


def input_vector(value):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("direction must contain three numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) or abs(x) > 1 for x in value):
        raise ValueError("direction values must be finite numbers in [-1, 1]")
    return [float(x) for x in value]


class Simulation:
    def __init__(self):
        self.controller = Controller(method="sdls")
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.direction = [0.0, 0.0, 0.0]
        self.enabled = False
        self.last_input = float("-inf")
        self.tick_times = deque(maxlen=1000)
        self.tick_ms = deque(maxlen=1000)
        self.overruns = 0
        self.fault = None
        self.thread = threading.Thread(target=self.run, name="kinematic-simulator", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)

    def set_input(self, body):
        if set(body) != {"direction", "enabled"} or type(body["enabled"]) is not bool:
            raise ValueError("expected direction and boolean enabled")
        direction = input_vector(body["direction"])
        with self.lock:
            self.direction = direction
            self.enabled = body["enabled"] and self.fault is None
            self.last_input = time.monotonic()

    def configure(self, body):
        if not body or not set(body) <= {"method", "elbow_weight"}:
            raise ValueError("expected method and/or elbow_weight")
        if "method" in body and body["method"] not in ("dls", "sdls"):
            raise ValueError("method must be dls or sdls")
        if "elbow_weight" in body:
            x = body["elbow_weight"]
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1:
                raise ValueError("elbow_weight must be in [0, 1]")
        with self.lock:
            self.controller.configure(**body)

    def reset(self):
        with self.lock:
            self.enabled = False
            self.direction = [0.0, 0.0, 0.0]
            self.last_input = float("-inf")
            self.fault = None
            self.controller.reset()
            self.tick_ms.clear()
            self.tick_times.clear()
            self.overruns = 0

    def snapshot(self):
        with self.lock:
            result = self.controller.state()
            stale = time.monotonic() - self.last_input > INPUT_TIMEOUT
            result.update(input_stale=stale, enabled=self.enabled and not stale and self.fault is None,
                          fault=self.fault, simulation_only=True, nominal_hz=1 / DT,
                          tick_ms=self.tick_ms[-1] if self.tick_ms else 0,
                          overruns=self.overruns)
            times = self.tick_times
            result["measured_hz"] = ((len(times) - 1) / (times[-1] - times[0])) if len(times) > 1 else 0
            if self.tick_ms:
                ordered = sorted(self.tick_ms)
                result["tick_p99_ms"] = ordered[math.ceil(0.99 * len(ordered)) - 1]
            else:
                result["tick_p99_ms"] = 0
            return result

    def run(self):
        deadline = time.monotonic()
        while not self.stop_event.is_set():
            start = time.monotonic()
            with self.lock:
                active = self.enabled and start - self.last_input <= INPUT_TIMEOUT and self.fault is None
                if start - deadline > INPUT_TIMEOUT:
                    # Suspension must not replay old motion on resume.
                    self.enabled = active = False
                    self.last_input = float("-inf")
                try:
                    self.controller.step(self.direction, active, DT)
                except Exception as exc:
                    # Hold until explicit reset; never keep executing a bad command.
                    self.fault = f"{type(exc).__name__}: {exc}"
                    self.enabled = False
                elapsed = (time.monotonic() - start) * 1000
                self.tick_ms.append(elapsed)
                self.tick_times.append(start)
                if elapsed > DT * 1000:
                    self.overruns += 1
            deadline += DT
            now = time.monotonic()
            if now > deadline:
                # No backlog/catch-up burst. Simulation time may lag wall time.
                deadline = now
            self.stop_event.wait(max(0, deadline - time.monotonic()))


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, simulation):
        self.simulation = simulation
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server: Server

    def log_message(self, *_args):
        pass

    def send_body(self, status, data, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def json_response(self, status, data):
        self.send_body(status, json.dumps(data, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/state":
            self.json_response(200, self.server.simulation.snapshot())
            return
        files = {"/": ("index.html", "text/html"), "/index.html": ("index.html", "text/html"),
                 "/app.js": ("app.js", "text/javascript"), "/styles.css": ("styles.css", "text/css")}
        if path not in files:
            self.json_response(404, {"error": "not found"})
            return
        name, mime = files[path]
        self.send_body(200, (STATIC / name).read_bytes(), mime + "; charset=utf-8")

    def do_POST(self):
        port = self.server.server_address[1]
        origin = self.headers.get("Origin")
        if origin is not None and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
            self.json_response(403, {"error": "cross-origin control is not allowed"})
            return
        if self.headers.get_content_type() != "application/json":
            self.json_response(415, {"error": "application/json is required"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("invalid request size")
            def reject_constant(value):
                raise ValueError(f"non-finite JSON value: {value}")
            body = json.loads(self.rfile.read(length), parse_constant=reject_constant)
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            path = urlsplit(self.path).path
            if path == "/api/input":
                self.server.simulation.set_input(body)
            elif path == "/api/config":
                self.server.simulation.configure(body)
            elif path == "/api/reset":
                if body:
                    raise ValueError("reset body must be empty")
                self.server.simulation.reset()
            else:
                self.json_response(404, {"error": "not found"})
                return
            self.json_response(200, {"ok": True})
        except (ValueError, TypeError, UnicodeDecodeError) as exc:
            self.json_response(400, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description="Gamepad arm teleoperation - local simulation only")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    simulation = Simulation()
    server = Server(("127.0.0.1", args.port), simulation)
    simulation.start()
    print(f"Arm Teleop Lab: http://127.0.0.1:{args.port}  (simulation only)", flush=True)
    print("Hold gamepad LB or keyboard Space to enable. Ctrl+C stops the server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        simulation.close()
        server.server_close()


if __name__ == "__main__":
    main()

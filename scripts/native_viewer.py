"""Run a local teleoperation server with a MuJoCo native display.

The native viewer displays a copy of the simulation data; control and physics
remain owned by Simulation. The browser must be focused to supply input.
"""
import argparse
import copy
from pathlib import Path
import sys
import threading
import time
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import mujoco.viewer

from arm_teleop.server import Server, Simulation
from arm_teleop.mujoco_backend import available_models


def copy_view_data(simulation, model, data):
    """Copy under the physics lock; return False if the robot was replaced."""
    with simulation.lock:
        if simulation.backend.model is not model:
            return False
        mujoco.mj_copyData(data, model, simulation.backend.data)
        return True


def show_viewer(simulation):
    """Reopen the passive viewer when the browser selects another robot."""
    while True:
        with simulation.lock:
            live_model = simulation.backend.model
            view_model = copy.copy(live_model)
            view_data = mujoco.MjData(view_model)
            mujoco.mj_copyData(view_data, live_model, simulation.backend.data)
        with mujoco.viewer.launch_passive(view_model, view_data,
                                           show_left_ui=False, show_right_ui=False) as viewer:
            while viewer.is_running():
                with viewer.lock():
                    current = copy_view_data(simulation, live_model, view_data)
                if not current:
                    break
                viewer.sync()
                time.sleep(1 / 30)
            else:
                return


def main():
    parser = argparse.ArgumentParser(description="MuJoCo viewer and browser teleoperation")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--robot", default="ur5e", help="robot profile name (default: ur5e)")
    parser.add_argument("--open-browser", action="store_true", help="open the browser control page")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    if args.robot not in available_models():
        parser.error(f"robot {args.robot!r} is unavailable; configured models: {', '.join(available_models())}")
    simulation = Simulation()
    simulation.configure({"robot": args.robot})
    server = Server(("127.0.0.1", args.port), simulation)
    simulation.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Open http://127.0.0.1:{args.port}/ and focus the browser to control.", flush=True)
    print("Native viewer shows the same simulation. Close it to stop this instance.", flush=True)
    try:
        if args.open_browser:
            webbrowser.open(f"http://127.0.0.1:{args.port}/")
        show_viewer(simulation)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
        simulation.close()


if __name__ == "__main__":
    main()

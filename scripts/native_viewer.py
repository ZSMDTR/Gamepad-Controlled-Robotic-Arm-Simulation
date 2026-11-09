"""Run the native MuJoCo practice window, optionally with the legacy web UI."""
import argparse
import copy
from pathlib import Path
import sys
import threading
import tempfile
import time
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import mujoco.viewer

from arm_teleop.server import DT, Server, Simulation, physics_steps
from arm_teleop.mujoco_backend import MujocoBackend, available_models
from arm_teleop.b601_scene import make_b601_scene_profile
from arm_teleop.world_axes import add_world_axes, update_tcp_axes
from optimized_algorithm.native_control import make_native_controller
from scripts.native_input import NativeInput, disabled
from scripts.window_layout import arrange_viewers, viewer_windows


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


def prepare_auxiliary_view(simulation):
    """Create an independent display copy; it never advances physics."""
    model = copy.copy(simulation.backend.model)
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, simulation.backend.data)
    viewer = mujoco.viewer.launch_passive(model, data,
                                          show_left_ui=False, show_right_ui=False)
    viewer.cam.lookat[:] = [.35, 0, .27]
    viewer.cam.distance = 1.3
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -55
    axes = add_world_axes(viewer.user_scn)
    if hasattr(viewer, 'set_texts'):
        viewer.set_texts((None, None, 'AUXILIARY VIEW - same simulation', ''))
    return viewer, model, data, axes


def sync_auxiliary_view(viewer, model, data, axes, simulation):
    """Refresh the display from the one authoritative simulation."""
    mujoco.mj_copyData(data, model, simulation.backend.data)
    update_tcp_axes(viewer.user_scn, axes, data.site_xpos[simulation.backend.tcp_id])
    viewer.sync()


def show_native_viewer(simulation):
    """Use the actual physics data, synchronizing viewer and physics access."""
    inputs = NativeInput()
    try:
        with mujoco.viewer.launch_passive(simulation.backend.model, simulation.backend.data,
                                          show_left_ui=True, show_right_ui=True) as viewer:
            simulation.native_hold_on_release = True
            if simulation.backend.name == 'b601_rs':
                viewer.cam.lookat[:] = [.2, 0, .28]
                viewer.cam.distance = 2.0
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
            axes = add_world_axes(viewer.user_scn)
            if hasattr(viewer, 'set_texts'):
                viewer.set_texts((None, None, 'MAIN VIEW - control', ''))
            main_windows = viewer_windows()
            auxiliary, aux_model, aux_data, aux_axes = prepare_auxiliary_view(simulation)
            arrange_viewers(main_windows, viewer_windows())
            print('Two MuJoCo windows show the same simulation. Close the auxiliary view independently.', flush=True)
            print('Focus MuJoCo. Shift + W/S: forward/back; A/D: left/right; Q/E: up/down.', flush=True)
            print('Shift + I/K, J/L, U/O: rotate about world X/Y/Z; Z/X: gripper.', flush=True)
            print('Gamepad: hold LB, move one stick axis; hold RB for rotation; A/B: gripper.', flush=True)
            try:
                deadline = time.monotonic()
                next_sync = deadline
                while viewer.is_running():
                    simulation.set_input(inputs.poll())
                    start = time.monotonic()
                    simulation.tick(start, deadline)
                    if start >= next_sync:
                        update_tcp_axes(viewer.user_scn, axes,
                                        simulation.backend.data.site_xpos[simulation.backend.tcp_id])
                        viewer.sync()
                        if auxiliary is not None:
                            if auxiliary.is_running():
                                sync_auxiliary_view(auxiliary, aux_model, aux_data, aux_axes, simulation)
                            else:
                                auxiliary = None
                        next_sync = start + 1 / 30
                    deadline += DT
                    now = time.monotonic()
                    if now > deadline:
                        deadline = now
                    time.sleep(max(0, deadline - time.monotonic()))
            finally:
                if auxiliary is not None:
                    auxiliary.close()
                simulation.set_input(disabled())
                simulation.close()
    finally:
        inputs.close()


def main():
    parser = argparse.ArgumentParser(description="MuJoCo native teleoperation")
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
    if not args.open_browser:
        with tempfile.TemporaryDirectory(prefix='b601-scene-') as scene_dir:
            if args.robot == 'b601_rs':
                profile = Path(__file__).resolve().parent.parent / 'robot_models' / 'b601_rs' / 'profile.json'
                scene_profile = make_b601_scene_profile(profile, scene_dir)
                simulation.backend = MujocoBackend('b601_rs', profile_path=scene_profile)
                simulation.physics_steps = physics_steps(simulation.backend.timestep)
                simulation.gripper_target = simulation.backend.state()['gripper_position']
            simulation.method = 'constrained_dls'
            simulation.elbow_weight = 0.
            simulation.controller = make_native_controller(simulation.backend.spec)
            try:
                show_native_viewer(simulation)
            except KeyboardInterrupt:
                pass
        return
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

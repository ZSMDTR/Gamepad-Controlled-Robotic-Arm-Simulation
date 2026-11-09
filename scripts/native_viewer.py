"""Run the native MuJoCo practice window, optionally with the legacy web UI."""
import argparse
import copy
import json
from pathlib import Path
import sys
import threading
import tempfile
import time
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import mujoco.viewer

from arm_teleop.server import DT, Server, Simulation, leader_configuration, physics_steps
from arm_teleop.mujoco_backend import MujocoBackend, available_models
from arm_teleop.b601_scene import make_b601_scene_profile
from arm_teleop.recorder import EpisodeRecorder, recording_metadata
from arm_teleop.world_axes import add_world_axes, update_tcp_axes
from optimized_algorithm.native_control import make_native_controller
from scripts.native_input import NativeInput, disabled
from scripts.window_layout import arrange_viewers, viewer_windows

RECORDING_DIR = Path(__file__).resolve().parent.parent / 'recordings'


def finish_recording(simulation, outcome, reason):
    """Close the current episode and report its actual on-disk outcome."""
    recorder = simulation.recorder
    if recorder is None:
        return 'NOT RECORDING'
    simulation.recorder = None
    if recorder.steps == 0:
        outcome, reason = 'interrupted', 'empty_episode'
    try:
        recorder.finish(outcome, reason)
    except Exception as exc:
        print(f'Recording incomplete: {recorder.path} ({type(exc).__name__}: {exc})', flush=True)
        return 'ERROR - see terminal'
    print(f'Episode saved ({outcome}, {recorder.steps} steps): {recorder.path}', flush=True)
    label = {'success': 'OK', 'failure': 'FAIL', 'interrupted': 'INTERRUPTED'}[outcome]
    return f'SAVED {label}: {recorder.steps}'


def set_recording_text(viewer, name, status):
    if viewer is not None and viewer.is_running() and hasattr(viewer, 'set_texts'):
        viewer.set_texts((None, None, f'{name} | {status}', 'F8 start | F9 OK | F10 fail'))


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


def show_native_viewer(simulation, reader=None, leader_config=None):
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
            main_windows = viewer_windows()
            auxiliary, aux_model, aux_data, aux_axes = prepare_auxiliary_view(simulation)
            arrange_viewers(main_windows, viewer_windows())
            scene = 'b601_practice' if simulation.backend.name == 'b601_rs' else 'robot_only'
            metadata = recording_metadata(simulation, scene, DT)
            if reader is not None:
                metadata['leader_calibration'] = leader_config
            record_status = 'READY'
            set_recording_text(viewer, 'MAIN VIEW', record_status)
            set_recording_text(auxiliary, 'AUX VIEW', record_status)
            print('Two MuJoCo windows show the same simulation. Close the auxiliary view independently.', flush=True)
            print('Focus MuJoCo. Shift + W/S: forward/back; A/D: left/right; Q/E: up/down.', flush=True)
            print('Shift + I/K, J/L, U/O: rotate about world X/Y/Z; Z/X: gripper.', flush=True)
            print('Gamepad: hold LB, move one stick axis; hold RB for rotation; A/B: gripper.', flush=True)
            print(f'F8 start recording; F9 save success; F10 save failure. Data: {RECORDING_DIR}', flush=True)
            if reader is not None:
                print('F11: toggle experimental leader following. Joint and gripper mappings require physical verification.', flush=True)
            try:
                deadline = time.monotonic()
                next_sync = deadline
                last_status_update = deadline
                suppress_manual = False
                while viewer.is_running():
                    sample = inputs.poll()
                    if (reader is not None and simulation.leader_active and
                            not inputs.record_commands.focused):
                        simulation.stop_leader('FOCUS LOST - PRESS F11')
                        suppress_manual = True
                    for command in inputs.pop_commands():
                        if command == 'start' and simulation.recorder is None:
                            try:
                                simulation.recorder = EpisodeRecorder(RECORDING_DIR, metadata)
                                simulation.recording_error = None
                                record_status = 'REC'
                                print(f'Recording: {simulation.recorder.path}', flush=True)
                            except Exception as exc:
                                record_status = 'ERROR - see terminal'
                                print(f'Cannot start recording: {type(exc).__name__}: {exc}', flush=True)
                        elif command in ('success', 'failure'):
                            record_status = finish_recording(simulation, command, 'user_label')
                        elif command == 'leader_toggle' and reader is not None:
                            if simulation.leader_active:
                                simulation.stop_leader()
                            else:
                                simulation.start_leader(reader.latest(), leader_config, time.monotonic())
                            suppress_manual = True
                            print(f'Leader: {simulation.leader_status}', flush=True)
                        set_recording_text(viewer, 'MAIN VIEW', record_status)
                        set_recording_text(auxiliary, 'AUX VIEW', record_status)
                    if simulation.leader_active:
                        simulation.set_leader_sample(reader.latest(), time.monotonic())
                        sample = disabled()
                    elif suppress_manual:
                        if not sample['enabled']:
                            suppress_manual = False
                        sample = disabled()
                    simulation.set_input(sample)
                    start = time.monotonic()
                    was_leader = simulation.leader_active
                    simulation.tick(start, deadline)
                    if was_leader and not simulation.leader_active:
                        suppress_manual = True
                    if simulation.recorder is not None and simulation.recording_error:
                        print(f'Recording stopped: {simulation.recording_error}', flush=True)
                        finish_recording(simulation, 'interrupted', 'recording_error')
                        record_status = 'ERROR - see terminal'
                        set_recording_text(viewer, 'MAIN VIEW', record_status)
                        set_recording_text(auxiliary, 'AUX VIEW', record_status)
                    if start - last_status_update >= .5:
                        if simulation.recorder is not None:
                            record_status = f'REC {simulation.recorder.steps}'
                        status = record_status if reader is None else f'{record_status} | LEADER {simulation.leader_status}'
                        set_recording_text(viewer, 'MAIN VIEW', status)
                        set_recording_text(auxiliary, 'AUX VIEW', status)
                        last_status_update = start
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
                if simulation.recorder is not None:
                    finish_recording(simulation, 'interrupted', 'viewer_closed')
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
    parser.add_argument("--leader-port", help="experimental read-only leader serial port, e.g. COM7")
    parser.add_argument("--leader-config", type=Path, help="JSON mapping for experimental leader following")
    args = parser.parse_args()
    if args.leader_config and not args.leader_port:
        parser.error('--leader-config requires --leader-port')
    if args.open_browser and args.leader_port:
        parser.error('--leader-port is supported in the native viewer only')
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
                if args.leader_port:
                    from arm_teleop.leader_serial import LeaderReader
                    config = {'joint_indices': list(range(6)), 'joint_scales': [1., 1., -1., -1., -1., 1.], 'gripper_scale': 1.2}
                    try:
                        if args.leader_config:
                            with args.leader_config.open(encoding='utf-8') as stream:
                                config = json.load(stream)
                        config = leader_configuration(config)
                    except (OSError, ValueError) as exc:
                        parser.error(f'invalid leader configuration: {exc}')
                    reader = LeaderReader(args.leader_port)
                    try:
                        reader.start()
                        show_native_viewer(simulation, reader, config)
                    finally:
                        reader.close()
                else:
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

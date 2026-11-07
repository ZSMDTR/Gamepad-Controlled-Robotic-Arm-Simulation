"""Replay the same commands on two MuJoCo models and compare pose controllers.

This is an offline, single-process simulation benchmark. It does not test
browser input latency, collisions, physical hardware, or hard real-time behavior.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm_teleop.mujoco_backend import MujocoBackend
from arm_teleop.pose_control import PoseController, rotation_exp, rotation_log


DT = .01
STEPS = 800


def command(step):
    """Eight seconds of identical world-frame input, including reversals."""
    return (
        ([.7, 0, 0], [0, 0, 0]),
        ([0, -.7, 0], [0, 0, .5]),
        ([0, 0, .6], [.5, 0, 0]),
        ([-.7, 0, 0], [0, -.5, 0]),
        ([0, .7, -.4], [0, 0, -.5]),
        ([0, 0, 0], [0, 0, 0]),
        ([.6, .4, 0], [.3, -.3, 0]),
        ([0, 0, 0], [0, 0, 0]),
    )[step // 100]


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def benchmark(robot, method, limit_scale=1.):
    backend = MujocoBackend(robot)
    spec = dict(backend.spec, mode=method)
    # A second run deliberately tightens reference limits to exercise
    # saturation; both methods receive exactly the same limits.
    spec['velocity_limit'] = spec['velocity_limit'] * limit_scale
    spec['acceleration_limit'] = spec['acceleration_limit'] * limit_scale
    controller = PoseController(spec)
    initial = backend.state()
    physics_steps = round(DT / backend.timestep)
    if physics_steps < 1 or not np.isclose(physics_steps * backend.timestep, DT):
        raise ValueError(f'{robot}: physics timestep does not divide {DT} seconds')
    ideal_position = initial['tcp'].copy()
    ideal_rotation = initial['rotation'].copy()
    solve_times, cycle_times = [], []
    bounded_pos, bounded_rot, ideal_pos, ideal_rot = [], [], [], []
    peak_command, peak_actual = 0., 0.
    constraints = Counter()
    started = time.perf_counter()
    for i in range(STEPS):
        cycle_start = time.perf_counter_ns()
        state = backend.state()
        translation, rotation = (np.asarray(x) for x in command(i))
        # Input components in this sequence are bounded to norm <= 1.
        ideal_position += translation * controller.cartesian_speed * DT
        ideal_rotation = rotation_exp(rotation * controller.angular_speed * DT) @ ideal_rotation
        result = controller.step(state['q'], state['qvel'], state['tcp'], state['rotation'],
                                 state['jacobian'], translation, rotation, True, DT)
        solve_times.append(result['solve_ms'])
        bounded_pos.append(result['error_m'])
        bounded_rot.append(result['error_rad'])
        ideal_pos.append(np.linalg.norm(ideal_position - state['tcp']))
        ideal_rot.append(np.linalg.norm(rotation_log(ideal_rotation @ state['rotation'].T)))
        peak_command = max(peak_command, float(np.max(np.abs(result['qdot_cmd']))))
        peak_actual = max(peak_actual, float(np.max(np.abs(state['qvel']))))
        constraints.update(set(result['constraints']))
        backend.command(result['q_ref'], initial['gripper_position'])
        backend.step(physics_steps, return_state=False)
        cycle_times.append((time.perf_counter_ns() - cycle_start) / 1e6)
    elapsed = time.perf_counter() - started
    return {
        'robot': robot, 'method': method, 'reference_limit_scale': limit_scale,
        'steps': STEPS, 'control_dt_s': DT, 'simulated_s': float(backend.data.time),
        'wall_s': elapsed, 'simulated_seconds_per_wall_second': float(backend.data.time) / elapsed,
        'controller_ms': {'mean': float(np.mean(solve_times)), 'p95': float(np.percentile(solve_times, 95)),
                          'p99': float(np.percentile(solve_times, 99)), 'max': float(max(solve_times))},
        'offline_cycle_ms': {'mean': float(np.mean(cycle_times)), 'p95': float(np.percentile(cycle_times, 95)),
                             'p99': float(np.percentile(cycle_times, 99)), 'max': float(max(cycle_times)),
                             'over_10ms_count': sum(t > DT * 1000 for t in cycle_times)},
        'bounded_target_rms': {'position_mm': rms(bounded_pos) * 1000, 'orientation_rad': rms(bounded_rot)},
        'common_unbounded_input_target_rms': {'position_mm': rms(ideal_pos) * 1000, 'orientation_rad': rms(ideal_rot)},
        'peak_abs_commanded_joint_velocity_rad_s': peak_command,
        'peak_abs_simulated_joint_velocity_rad_s': peak_actual,
        'constraint_event_steps': dict(sorted(constraints.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='optional JSON output path')
    args = parser.parse_args()
    results = [benchmark(robot, method, limit_scale)
               for limit_scale in (1., .25) for robot in ('ur5e', 'panda')
               for method in ('dls', 'constrained_dls')]
    report = {
        'scope': 'Offline MuJoCo replay; controller limits apply to references, not guaranteed physical limits. No collision or real-robot safety evaluation.',
        'timing_note': 'No pacing, GUI, browser, or network. Cycle time includes benchmark instrumentation, state reads, controller and MuJoCo steps; system load affects timings.',
        'tracking_note': 'Bounded target is controller-specific; common unbounded target integrates the same inputs from each model initial pose and can be unreachable. Both errors use measured simulated pose before physics advances.',
        'constraint_note': 'Counts are controller-reported reference constraint labels, counted once per control step; different solvers may report different labels. Scale 0.25 is a synthetic test with tighter velocity and acceleration limits, not the stock profile.',
        'results': results,
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + '\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()

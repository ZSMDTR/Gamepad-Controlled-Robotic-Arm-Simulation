"""Deterministic solver/controller benchmark; excludes browser/network/actuator delay."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm_teleop.controller import Controller


def benchmark(method, steps, dt):
    ctrl = Controller(method)
    for _ in range(100):
        ctrl.step([.2, .2, .1], True, dt)
    ctrl.reset()
    durations, errors, speeds = [], [], []
    capped = 0
    for i in range(steps):
        t = i * dt
        direction = [.65 * np.cos(t), .65 * np.sin(t), .3 * np.cos(.7 * t)]
        start = time.perf_counter_ns()
        state = ctrl.step(direction, True, dt)
        durations.append((time.perf_counter_ns() - start) / 1e6)
        errors.append(state["error_m"])
        speeds.append(max(abs(x) for x in state["joint_velocity"]))
        capped += int("target_lead" in state["constraints"])
    return dict(method=method, steps=steps, dt=dt, mean_ms=float(np.mean(durations)),
                p99_ms=float(np.percentile(durations, 99)), max_ms=max(durations),
                tracking_rms_mm=float(np.sqrt(np.mean(np.square(errors))) * 1000),
                peak_joint_speed_rad_s=max(speeds), target_capped_steps=capped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=.01)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or not 0 < args.dt <= .05:
        parser.error("steps must be positive and dt must be in (0, 0.05]")
    results = [benchmark(method, args.steps, args.dt) for method in ("dls", "sdls")]
    if args.json:
        print(json.dumps({"scope": "controller only; no hard-real-time guarantee", "results": results}, indent=2))
    else:
        print("method   mean(ms)   p99(ms)   max(ms)   RMS error(mm)   target cap steps")
        for r in results:
            print(f"{r['method']:6s} {r['mean_ms']:10.3f} {r['p99_ms']:9.3f} {r['max_ms']:9.3f} {r['tracking_rms_mm']:15.3f} {r['target_capped_steps']:18d}")
        print("Measured Python controller cost only; excludes browser, network and hardware.")
        print("Error uses the bounded reference, not the unrestricted integrated joystick trajectory.")


if __name__ == "__main__":
    main()

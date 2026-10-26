"""Position-only differential IK with bounded commands and posture preference."""

from __future__ import annotations

import time
import numpy as np

from .kinematics import ArmModel, vector
from .solvers import dls, sdls, nullspace_projector


class Controller:
    def __init__(self, method="sdls"):
        self.model = ArmModel()
        self.method = "sdls"
        self.elbow_weight = .35
        self.cartesian_speed = .12
        self.max_target_lead = .06
        self.gain = 8.0
        self.damping = .04
        self.configure(method=method)
        self.reset()

    def configure(self, method=None, elbow_weight=None):
        if method is not None and method not in ("dls", "sdls"):
            raise ValueError("method must be dls or sdls")
        if elbow_weight is not None and (isinstance(elbow_weight, bool) or not np.isfinite(elbow_weight) or not 0 <= elbow_weight <= 1):
            raise ValueError("elbow_weight must be finite and in [0, 1]")
        if method is not None:
            self.method = method
        if elbow_weight is not None:
            self.elbow_weight = float(elbow_weight)

    def reset(self):
        self.q = self.model.home.copy()
        self.qdot = np.zeros(6)
        self.points, self.jacobian = self.model.geometry(self.q)
        self.target = self.points[-1].copy()
        self.solve_ms = 0.0
        self.sigma_min = float(np.linalg.svd(self.jacobian, compute_uv=False)[-1])
        self.enabled = False
        self.constraints = []
        return self.state()

    def step(self, direction, enabled, dt):
        direction = vector(direction, 3, "direction")
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if not np.isfinite(dt) or not 0 < dt <= .05:
            raise ValueError("dt must be finite and in (0, 0.05]")
        if np.max(np.abs(direction)) > 1:
            raise ValueError("direction values must be in [-1, 1]")
        start = time.perf_counter()
        self.enabled = enabled
        self.constraints = []
        norm = np.linalg.norm(direction)
        if not enabled or norm < 1e-8:
            # Exact pause is a kinematic simulation feature, not a physical brake.
            self.qdot[:] = 0
            self.target = self.points[-1].copy()
            self.constraints = ["hold"]
            # Retain the last moving-step time; a paused tick is not an IK solve.
            return self.state()
        direction = direction / max(1.0, norm)
        target = self.target + direction * self.cartesian_speed * dt
        error = target - self.points[-1]
        length = np.linalg.norm(error)
        if length > self.max_target_lead:
            error *= self.max_target_lead / length
            target = self.points[-1] + error
            self.constraints.append("target_lead")

        # Compute one SVD and share it across solve, rank and posture projection.
        decomposition = np.linalg.svd(self.jacobian, full_matrices=False)
        self.sigma_min = float(decomposition[1][-1])
        step_error = error * min(1.0, self.gain * dt)
        if self.method == "sdls":
            dq = sdls(self.jacobian, step_error, gamma_max=.2, decomposition=decomposition)
        else:
            dq = dls(self.jacobian, step_error, self.damping, decomposition=decomposition)

        # A home-posture preference, weighted toward shoulder/elbow, not an
        # independently controllable geometric elbow target.
        posture = (self.model.home - self.q) * np.array([.3, .6, 1., .3, .3, .3])
        secondary = nullspace_projector(self.jacobian, decomposition) @ posture
        desired = dq / dt + self.elbow_weight * secondary
        vmax = self.model.velocity_limit
        amax = self.model.acceleration_limit
        limited = np.clip(desired, -vmax, vmax)
        if not np.allclose(limited, desired, rtol=0, atol=1e-12):
            self.constraints.append("velocity")

        # Conservative discrete braking envelope; stop before position limits.
        dist_low = np.maximum(0, self.q - self.model.lower)
        dist_high = np.maximum(0, self.model.upper - self.q)
        brake_low = np.maximum(0, np.sqrt((amax * dt) ** 2 + 2 * amax * dist_low) - amax * dt)
        brake_high = np.maximum(0, np.sqrt((amax * dt) ** 2 + 2 * amax * dist_high) - amax * dt)
        limited = np.clip(limited, -brake_low, brake_high)
        velocity = np.clip(limited, self.qdot - amax * dt, self.qdot + amax * dt)
        if not np.allclose(velocity, limited, rtol=0, atol=1e-12):
            self.constraints.append("acceleration")
        q_next = self.q + velocity * dt
        clipped = np.clip(q_next, self.model.lower, self.model.upper)
        if not np.array_equal(q_next, clipped):
            self.constraints.append("position_hard_stop")
            velocity = (clipped - self.q) / dt
        if not np.isfinite(clipped).all():
            raise ValueError("solver returned a non-finite configuration")
        self.q = clipped
        self.qdot = velocity
        self.target = target
        self.points, self.jacobian = self.model.geometry(self.q)
        # Reapply the cap against the new TCP after constraint filtering.
        lead = self.target - self.points[-1]
        if np.linalg.norm(lead) > self.max_target_lead:
            self.target = self.points[-1] + lead * self.max_target_lead / np.linalg.norm(lead)
        self.solve_ms = (time.perf_counter() - start) * 1000
        return self.state()

    def state(self):
        return {
            "joints": self.q.tolist(), "joint_velocity": self.qdot.tolist(),
            "points": self.points.tolist(), "tcp": self.points[-1].tolist(),
            "target": self.target.tolist(),
            "error_m": float(np.linalg.norm(self.target - self.points[-1])),
            "solve_ms": self.solve_ms, "sigma_min": self.sigma_min,
            "method": self.method, "elbow_weight": self.elbow_weight,
            "enabled": self.enabled, "constraints": self.constraints.copy(),
            "joint_lower": self.model.lower.tolist(), "joint_upper": self.model.upper.tolist(),
        }

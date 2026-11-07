"""Model-independent, world-frame pose teleoperation using differential IK.

DLS is the 6D baseline. The experimental SDLS extends the original position
method to a scaled six-dimensional pose task using two three-dimensional
blocks; it is not the original paper's pose solver. The position-priority mode projects angular motion
into the exact instantaneous position nullspace. For the baseline modes,
joint clipping after solving can change task priorities. The optional
constrained DLS mode incorporates velocity and reference bounds in a convex
quadratic solve; conflicting bounds relax the reference-lead constraint and
advance the reference without a jump. All limits apply to references, not a guarantee about physical robot motion.
"""
from __future__ import annotations

import time
import numpy as np


def _vec(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result


def _bounded(vector, bound):
    length = float(np.linalg.norm(vector))
    return vector * min(1.0, bound / length) if length > 0 else vector.copy()


def rotation_exp(rotvec):
    """SO(3) exponential; rotation vector in radians."""
    v = _vec(rotvec, 3, "rotation vector")
    angle = float(np.linalg.norm(v))
    k = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    if angle < 1e-8:
        return np.eye(3) + k + .5 * k @ k
    return np.eye(3) + np.sin(angle) / angle * k + ((1 - np.cos(angle)) / angle**2) * (k @ k)


def rotation_log(matrix):
    """Shortest world-frame rotation vector, stable near zero and pi.

    At exactly pi either axis sign represents the same rotation; the chosen
    sign can change at that intrinsic discontinuity.
    """
    r = np.asarray(matrix, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("rotation matrix must be finite and 3x3")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or abs(np.linalg.det(r) - 1) > 1e-5:
        raise ValueError("rotation matrix must be orthonormal with determinant 1")
    cosine = np.clip((np.trace(r) - 1) * .5, -1., 1.)
    angle = float(np.arccos(cosine))
    skew = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if angle < 1e-7:
        return .5 * skew
    if np.pi - angle < 1e-5:
        # Principal eigenvector of the symmetric component gives the axis.
        _, axes = np.linalg.eigh((r + r.T + 2 * np.eye(3)) * .25)
        axis = axes[:, -1]
        if np.dot(axis, skew) < 0:
            axis = -axis
        return axis * angle
    return angle / (2 * np.sin(angle)) * skew


def _dls(j, velocity, damping):
    u, s, vh = np.linalg.svd(j, full_matrices=False)
    return vh.T @ ((s / (s * s + damping * damping)) * (u.T @ velocity))


def _projector(j):
    _, s, vh = np.linalg.svd(j, full_matrices=False)
    cutoff = max(1e-10, s[0] * 1e-8) if s.size else 1e-10
    basis = vh[s > cutoff]
    return np.eye(j.shape[1]) - basis.T @ basis


def _box_dls(j, target, damping, lower, upper, center):
    """Minimize damped task error with box bounds by a feasible active set.

    Returns (velocity, converged). The quadratic is strictly convex because
    damping is positive. A bounded iteration count keeps control time finite.
    """
    h = j.T @ j + damping**2 * np.eye(j.shape[1])
    b = j.T @ target + damping**2 * center
    x = np.clip(np.zeros(j.shape[1]), lower, upper)
    tol = 1e-10
    active = (x <= lower + tol) | (x >= upper - tol)
    fixed = np.abs(upper - lower) <= tol
    for _ in range(50 + 10 * j.shape[1]):
        gradient = h @ x - b
        free = ~active
        direction = np.zeros_like(x)
        if np.any(free):
            direction[free] = np.linalg.solve(h[np.ix_(free, free)], -gradient[free])
        if np.max(np.abs(direction), initial=0.) <= tol:
            # Release the active bound with the largest KKT violation.
            violation = np.zeros_like(x)
            at_lower = active & ~fixed & (x <= lower + tol)
            at_upper = active & ~fixed & (x >= upper - tol)
            violation[at_lower] = np.maximum(0., -gradient[at_lower])
            violation[at_upper] = np.maximum(0., gradient[at_upper])
            if np.max(violation, initial=0.) <= tol:
                return x, True
            active[np.argmax(violation)] = False
            continue
        alpha = 1.
        positive, negative = direction > tol, direction < -tol
        if np.any(positive):
            alpha = min(alpha, float(np.min((upper[positive] - x[positive]) / direction[positive])))
        if np.any(negative):
            alpha = min(alpha, float(np.min((lower[negative] - x[negative]) / direction[negative])))
        x = np.clip(x + max(0., alpha) * direction, lower, upper)
        if alpha < 1. - tol:
            active |= free & ((x <= lower + tol) | (x >= upper - tol))
    return x, False


def sdls_pose(jacobian, error, gamma_max=.2):
    """Experimental scaled pose SDLS, extending the paper to 3D blocks.

    The 6D version treats linear displacement and length-scaled rotation as
    two three-dimensional blocks; choose length_scale before calling. A 3D
    input has the original single-target definition. Error and the result
    are increments, not velocities. This heuristic is not a constraint solver.
    """
    j = np.asarray(jacobian, dtype=float)
    e = np.asarray(error, dtype=float)
    if j.ndim != 2 or j.shape[0] not in (3, 6) or j.shape[1] < 1 or e.shape != (j.shape[0],):
        raise ValueError('SDLS requires a 3xn or 6xn Jacobian and matching increment')
    if not np.isfinite(j).all() or not np.isfinite(e).all() or not np.isfinite(gamma_max) or gamma_max <= 0:
        raise ValueError('SDLS inputs must be finite and gamma_max positive')
    u, s, vh = np.linalg.svd(j, full_matrices=False)
    threshold = max(1e-10, float(s[0]) * 1e-8)
    rho = sum(np.linalg.norm(j[start:start + 3], axis=0) for start in range(0, j.shape[0], 3))
    result = np.zeros(j.shape[1])
    for i, sigma in enumerate(s):
        if sigma <= threshold:
            continue
        v = vh[i]
        ni = sum(np.linalg.norm(u[start:start + 3, i]) for start in range(0, j.shape[0], 3))
        mi = float(np.abs(v) @ rho) / sigma
        bound = gamma_max * min(1., ni / mi) if mi > 0 else gamma_max
        component = v * (u[:, i] @ e) / sigma
        peak = np.max(np.abs(component), initial=0.)
        if peak:
            component *= min(1., bound / peak)
        result += component
    peak = np.max(np.abs(result), initial=0.)
    return result * min(1., gamma_max / peak) if peak else result


class PoseController:
    """Finite-step 6D reference generator with bounded target lead.

    spec keys: lower, upper, velocity_limit, acceleration_limit, optional
    home, mode ('dls', 'sdls', 'position_priority' or 'constrained_dls'), and tuning keys below.
    step() consumes measured state and a world-frame geometric Jacobian.
    The returned q_ref is a position actuator reference, not actual q.
    """

    def __init__(self, spec):
        self.lower = np.asarray(spec['lower'], dtype=float)
        if self.lower.ndim != 1 or self.lower.size == 0 or not np.isfinite(self.lower).all():
            raise ValueError('lower must be a finite nonempty vector')
        self.n = self.lower.size
        self.upper = _vec(spec['upper'], self.n, 'upper').copy()
        self.vmax = _vec(spec['velocity_limit'], self.n, 'velocity_limit').copy()
        self.amax = _vec(spec['acceleration_limit'], self.n, 'acceleration_limit').copy()
        self.home = _vec(spec.get('home', np.clip(np.zeros(self.n), self.lower, self.upper)), self.n, 'home').copy()
        if np.any(self.lower >= self.upper) or np.any(self.vmax <= 0) or np.any(self.amax <= 0):
            raise ValueError('invalid position, velocity or acceleration limits')
        if np.any(self.home < self.lower) or np.any(self.home > self.upper):
            raise ValueError('home outside limits')
        defaults = dict(position_gain=6., orientation_gain=5., cartesian_speed=.12,
                        angular_speed=.7, length_scale=.25, damping=.04,
                        max_position_lead=.04, max_orientation_lead=.35,
                        max_joint_reference_lead=.12, posture_gain=.08,
                        sdls_gamma=.2)
        for key, default in defaults.items():
            value = float(spec.get(key, default))
            if not np.isfinite(value) or value < 0 or (key != 'posture_gain' and value == 0):
                raise ValueError(f'{key} must be positive and finite')
            setattr(self, key, value)
        self.set_mode(spec.get('mode', 'dls'))
        self.reset()

    def set_mode(self, mode):
        if mode not in ('dls', 'sdls', 'position_priority', 'constrained_dls'):
            raise ValueError("mode must be 'dls', 'sdls', 'position_priority' or 'constrained_dls'")
        self.mode = mode

    def reset(self):
        self.target_position = None
        self.target_rotation = None
        self.q_ref = None
        self.velocity = np.zeros(self.n)
        self.was_enabled = False

    def step(self, q, qvel, tcp, rotation, jacobian, translation, rotation_input, enabled, dt):
        start = time.perf_counter()
        q = _vec(q, self.n, 'q')
        qvel = _vec(qvel, self.n, 'qvel')
        tcp = _vec(tcp, 3, 'tcp')
        rotation = np.asarray(rotation, dtype=float)
        rotation_log(rotation)  # validates the rotation before mutating targets
        jacobian = np.asarray(jacobian, dtype=float)
        if jacobian.shape != (6, self.n) or not np.isfinite(jacobian).all():
            raise ValueError('jacobian must be finite and 6xn')
        translation = _vec(translation, 3, 'translation')
        rotation_input = _vec(rotation_input, 3, 'rotation_input')
        if np.any(np.abs(translation) > 1) or np.any(np.abs(rotation_input) > 1):
            raise ValueError('input axes must be in [-1, 1]')
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError('enabled must be boolean')
        if not np.isfinite(dt) or not 0 < dt <= .05:
            raise ValueError('dt must be finite and in (0, 0.05]')
        constraints = []
        relax_reference_lead = False
        if self.q_ref is None:
            self.q_ref = np.clip(q.copy(), self.lower, self.upper)
            self.target_position = tcp.copy()
            self.target_rotation = rotation.copy()

        if enabled:
            if not self.was_enabled:
                # Start from measured pose, avoiding target jumps after a pause.
                self.target_position = tcp.copy()
                self.target_rotation = rotation.copy()
            self.target_position += _bounded(translation, 1.) * self.cartesian_speed * dt
            self.target_rotation = rotation_exp(_bounded(rotation_input, 1.) * self.angular_speed * dt) @ self.target_rotation
            e_pos = self.target_position - tcp
            e_rot = rotation_log(self.target_rotation @ rotation.T)
            if np.linalg.norm(e_pos) > self.max_position_lead:
                e_pos = _bounded(e_pos, self.max_position_lead)
                self.target_position = tcp + e_pos
                constraints.append('position_target_lead')
            if np.linalg.norm(e_rot) > self.max_orientation_lead:
                e_rot = _bounded(e_rot, self.max_orientation_lead)
                self.target_rotation = rotation_exp(e_rot) @ rotation
                constraints.append('rotation_target_lead')
            jpos, jrot = jacobian[:3], jacobian[3:]
            vpos = _bounded(self.position_gain * e_pos, self.cartesian_speed)
            vrot = _bounded(self.orientation_gain * e_rot, self.angular_speed)
            if self.mode in ('dls', 'sdls', 'constrained_dls'):
                jscaled = np.vstack((jpos, self.length_scale * jrot))
                vscaled = np.r_[vpos, self.length_scale * vrot]
                desired = (_dls(jscaled, vscaled, self.damping) if self.mode in ('dls', 'constrained_dls')
                           else sdls_pose(jscaled, vscaled * dt, self.sdls_gamma) / dt)
                null = _projector(jscaled)
            else:
                pos = _dls(jpos, vpos, self.damping)
                p = _projector(jpos)
                # Damped projected task; exact projector prevents first-order
                # interference with position before subsequent joint clipping.
                jr = self.length_scale * jrot @ p
                desired = pos + p @ _dls(jr, self.length_scale * (vrot - jrot @ pos), self.damping)
                null = _projector(np.vstack((jpos, jrot)))
            desired += self.posture_gain * (null @ (self.home - q))
            if self.mode == 'constrained_dls':
                # Include the same reference and braking bounds used below in
                # the solve, together with the per-tick acceleration interval.
                dist_lo = np.maximum(0., q - self.lower)
                dist_hi = np.maximum(0., self.upper - q)
                v_lo = np.maximum(0., np.sqrt((self.amax * dt)**2 + 2 * self.amax * dist_lo) - self.amax * dt)
                v_hi = np.maximum(0., np.sqrt((self.amax * dt)**2 + 2 * self.amax * dist_hi) - self.amax * dt)
                ref_lo = np.maximum(self.lower, q - self.max_joint_reference_lead)
                ref_hi = np.minimum(self.upper, q + self.max_joint_reference_lead)
                ref_dist_lo = np.maximum(0., self.q_ref - ref_lo)
                ref_dist_hi = np.maximum(0., ref_hi - self.q_ref)
                ref_v_lo = np.maximum(0., np.sqrt((self.amax * dt)**2 + 2 * self.amax * ref_dist_lo) - self.amax * dt)
                ref_v_hi = np.maximum(0., np.sqrt((self.amax * dt)**2 + 2 * self.amax * ref_dist_hi) - self.amax * dt)
                lo = np.maximum.reduce((-self.vmax, -v_lo, -ref_v_lo, self.velocity - self.amax * dt,
                                        (ref_lo - self.q_ref) / dt))
                hi = np.minimum.reduce((self.vmax, v_hi, ref_v_hi, self.velocity + self.amax * dt,
                                        (ref_hi - self.q_ref) / dt))
                if np.any(lo > hi):
                    # Preserve a continuous reference when acceleration and
                    # lead/braking intervals cannot all be satisfied. The
                    # lead/braking envelopes are then advisory for this tick.
                    constraints.extend(('infeasible_velocity_bounds', 'reference_lead_and_braking_relaxed'))
                    relax_reference_lead = True
                else:
                    desired, converged = _box_dls(jscaled, vscaled, self.damping, lo, hi,
                                                   self.posture_gain * (self.home - q))
                    if not converged:
                        constraints.append('solver_iteration_limit')
                    if np.any(np.isclose(desired, lo, atol=1e-9)) or np.any(np.isclose(desired, hi, atol=1e-9)):
                        constraints.append('constrained_velocity')
            if np.any(np.abs(desired) > self.vmax):
                constraints.append('velocity')
            desired = np.clip(desired, -self.vmax, self.vmax)
            # Velocity braking envelope based on measured position. The
            # subsequent acceleration cap can still be infeasible near a hard
            # stop; physical limits must also be enforced by the backend.
            dist_lo = np.maximum(0., q - self.lower)
            dist_hi = np.maximum(0., self.upper - q)
            v_lo = np.maximum(0., np.sqrt((self.amax * dt)**2 + 2 * self.amax * dist_lo) - self.amax * dt)
            v_hi = np.maximum(0., np.sqrt((self.amax * dt)**2 + 2 * self.amax * dist_hi) - self.amax * dt)
            limited = np.clip(desired, -v_lo, v_hi)
            if not np.allclose(limited, desired):
                constraints.append('braking')
            desired = limited
        else:
            # Deadman release: command deceleration, and re-anchor Cartesian
            # target so old motion is never replayed on the next activation.
            self.target_position = tcp.copy()
            self.target_rotation = rotation.copy()
            desired = np.zeros(self.n)
            constraints.append('hold')
        velocity = np.clip(desired, self.velocity - self.amax * dt, self.velocity + self.amax * dt)
        if not np.allclose(velocity, desired):
            constraints.append('acceleration')
        previous_velocity = self.velocity.copy()
        self.velocity = velocity
        previous_ref = self.q_ref.copy()
        requested_ref = previous_ref + velocity * dt
        if self.mode == 'constrained_dls' and not enabled:
            ref_lo = np.maximum(self.lower, q - self.max_joint_reference_lead)
            ref_hi = np.minimum(self.upper, q + self.max_joint_reference_lead)
            if np.any(requested_ref < ref_lo) or np.any(requested_ref > ref_hi):
                # A moved lead window can conflict with deadman deceleration;
                # preserve the continuous reference instead of snapping to it.
                constraints.extend(('deadman_reference_lead_conflict', 'reference_lead_and_braking_relaxed'))
                relax_reference_lead = True
        if relax_reference_lead:
            # Absolute model joint limits have priority over velocity continuity.
            # Unlike clipping to the changing measured-position lead interval,
            # these fixed limits cannot move the reference discontinuously.
            self.q_ref = np.clip(requested_ref, self.lower, self.upper)
            velocity = (self.q_ref - previous_ref) / dt
            self.velocity = velocity
            if np.any(np.abs(velocity - previous_velocity) > self.amax * dt + 1e-10):
                constraints.append('acceleration_relaxed_at_joint_limit')
        else:
            self.q_ref = np.clip(requested_ref, np.maximum(self.lower, q - self.max_joint_reference_lead),
                                 np.minimum(self.upper, q + self.max_joint_reference_lead))
        if not np.allclose(self.q_ref, requested_ref):
            constraints.append('joint_reference_lead_or_limit')
        self.was_enabled = bool(enabled)
        error_position = self.target_position - tcp
        error_rotation = rotation_log(self.target_rotation @ rotation.T)
        return dict(q_ref=self.q_ref.copy(), qdot_cmd=velocity.copy(),
                    target_position=self.target_position.copy(), target_rotation=self.target_rotation.copy(),
                    error_m=float(np.linalg.norm(error_position)), error_rad=float(np.linalg.norm(error_rotation)),
                    constraints=constraints, mode=self.mode, solve_ms=(time.perf_counter()-start)*1000.)

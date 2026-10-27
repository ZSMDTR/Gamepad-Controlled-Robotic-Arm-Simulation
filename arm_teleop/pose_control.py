"""Model-independent, world-frame pose teleoperation using differential IK.

DLS is the 6D baseline. The experimental SDLS extends the original position
method to a scaled six-dimensional pose task using two three-dimensional
blocks; it is not the original paper's pose solver. The position-priority mode projects angular motion
into the exact instantaneous position nullspace. Joint clipping is applied
*after* solving and can change task priorities. This is not a QP or SDLS.
All limits apply to references, not a guarantee about physical robot motion.
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
    home, mode ('dls', 'sdls' or 'position_priority'), and tuning keys below.
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
        if mode not in ('dls', 'sdls', 'position_priority'):
            raise ValueError("mode must be 'dls', 'sdls' or 'position_priority'")
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
            if self.mode in ('dls', 'sdls'):
                jscaled = np.vstack((jpos, self.length_scale * jrot))
                vscaled = np.r_[vpos, self.length_scale * vrot]
                desired = (_dls(jscaled, vscaled, self.damping) if self.mode == 'dls'
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
        self.velocity = velocity
        requested_ref = self.q_ref + velocity * dt
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

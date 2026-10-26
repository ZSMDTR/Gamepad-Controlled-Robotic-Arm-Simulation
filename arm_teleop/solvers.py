"""Original implementation of DLS and single-TCP Buss/Kim SDLS.

SDLS follows the hard ClampMaxAbs in paper equations (3)-(4), rather than
the smooth rescaling in the authors' downloadable 2004 C++ demo. The
controller's fixed target lead cap is an engineering choice; it does not
reproduce the paper's adaptive per-effector target-distance heuristic.
"""

from __future__ import annotations

import numpy as np


def _inputs(jacobian, error):
    j = np.asarray(jacobian, dtype=float)
    e = np.asarray(error, dtype=float)
    if j.ndim != 2 or min(j.shape) == 0 or e.shape != (j.shape[0],):
        raise ValueError("incompatible Jacobian and error shapes")
    if not np.isfinite(j).all() or not np.isfinite(e).all():
        raise ValueError("Jacobian and error must be finite")
    return j, e


def clamp_max_abs(value, bound):
    """Scale the whole vector, preserving direction (not componentwise clip)."""
    peak = float(np.max(np.abs(value), initial=0))
    return value * min(1.0, bound / peak) if peak > 0 else value.copy()


def dls(jacobian, error, damping=.04, decomposition=None):
    j, e = _inputs(jacobian, error)
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be positive and finite")
    u, s, vh = decomposition if decomposition is not None else np.linalg.svd(j, full_matrices=False)
    return vh.T @ ((s / (s * s + damping * damping)) * (u.T @ e))


def sdls(jacobian, error, gamma_max=.2, decomposition=None):
    """Paper SDLS for one 3-D position target, all joints rotational.

    For one end effector N_i = ||u_i|| = 1. M_i measures the sum of the
    individual joint contributions, before their cancellation. Very small
    singular values are truncated for numerical stability.
    """
    j, e = _inputs(jacobian, error)
    if j.shape[0] != 3:
        raise ValueError("SDLS implementation expects one XYZ target")
    if not np.isfinite(gamma_max) or gamma_max <= 0:
        raise ValueError("gamma_max must be positive and finite")
    u, s, vh = decomposition if decomposition is not None else np.linalg.svd(j, full_matrices=False)
    rho = np.linalg.norm(j, axis=0)
    threshold = max(1e-10, float(s[0]) * 1e-8)
    result = np.zeros(j.shape[1])
    for i, sigma in enumerate(s):
        if sigma <= threshold:
            continue
        v = vh[i]
        n_i = np.linalg.norm(u[:, i])
        m_i = float(np.abs(v) @ rho) / sigma
        gamma_i = gamma_max * min(1.0, n_i / m_i) if m_i > 0 else gamma_max
        contribution = v * float(u[:, i] @ e) / sigma
        result += clamp_max_abs(contribution, gamma_i)
    return clamp_max_abs(result, gamma_max)


def nullspace_projector(jacobian, decomposition=None):
    j = np.asarray(jacobian, dtype=float)
    if j.ndim != 2 or min(j.shape) == 0 or not np.isfinite(j).all():
        raise ValueError("Jacobian must be finite and non-empty")
    _, s, vh = decomposition if decomposition is not None else np.linalg.svd(j, full_matrices=False)
    basis = vh[s > max(1e-10, float(s[0]) * 1e-8)]
    return np.eye(j.shape[1]) - basis.T @ basis

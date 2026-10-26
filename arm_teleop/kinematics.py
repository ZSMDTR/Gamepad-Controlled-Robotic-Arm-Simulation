"""Explicit serial-chain geometry in metres/radians; no hardware-specific model."""

from __future__ import annotations

import numpy as np


def vector(value, size, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite vector of size {size}")
    return value


def rotation(axis, angle):
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


class ArmModel:
    """Six revolute joints: local-axis rotation, then a local link translation.

    Each row defines its axis in the preceding link frame. The resulting
    rotation also rotates that row's link offset. TCP is the final endpoint;
    joint 6 has an offset TCP and therefore can affect position.
    """

    def __init__(self):
        self.axes = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0],
                              [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        self.offsets = np.array([[0, 0, .15], [.25, 0, 0], [.23, 0, 0],
                                 [.08, 0, 0], [.08, 0, 0], [.07, 0, 0]], dtype=float)
        self.lower = np.array([-2.8, -1.5, -2.6, -2.8, -1.6, -2.8])
        self.upper = np.array([2.8, 1.2, 2.6, 2.8, 1.6, 2.8])
        self.home = np.array([0.0, -.65, 1.15, .2, -.5, .1])
        self.velocity_limit = np.full(6, .8)
        self.acceleration_limit = np.full(6, 2.5)

    def geometry(self, q):
        q = vector(q, 6, "q")
        frame = np.eye(3)
        points = np.zeros((7, 3))
        axes = np.empty((6, 3))
        for i in range(6):
            axes[i] = frame @ self.axes[i]
            frame = frame @ rotation(self.axes[i], q[i])
            points[i + 1] = points[i] + frame @ self.offsets[i]
        jacobian = np.cross(axes, points[-1] - points[:-1]).T
        return points, jacobian

    def fk(self, q):
        return self.geometry(q)[0]

    def jacobian(self, q):
        return self.geometry(q)[1]

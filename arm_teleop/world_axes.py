"""World-aligned direction arrows for the native MuJoCo viewer.

The first triad is an offset orientation guide, not the world origin.  The
smaller triad follows the TCP while remaining parallel to the world axes.
These are viewer-only decorations, so they cannot affect collisions.
"""
import mujoco
import numpy as np


_DIRECTIONS = np.eye(3)
_COLORS = np.array([[1., .16, .1, 1.], [.12, .95, .22, 1.],
                    [.16, .5, 1., 1.]], dtype=np.float32)
_LABELS = ('+X FRONT', '+Y LEFT', '+Z UP')
_GUIDE_ORIGIN = np.array([.42, -.36, .35])


def add_world_axes(scene):
    """Add a fixed guide and a TCP guide to ``viewer.user_scn``.

    Return the index used by :func:`update_tcp_axes`. Call once before the
    first viewer sync. The guide uses nine out of the scene's geom slots.
    """
    if scene.ngeom + 9 > scene.maxgeom:
        raise ValueError('MuJoCo viewer scene has insufficient room for world axes')
    first = scene.ngeom
    identity = np.eye(3).ravel()
    for i, direction in enumerate(_DIRECTIONS):
        arrow = scene.geoms[first + i]
        mujoco.mjv_initGeom(arrow, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                            np.zeros(3), identity, _COLORS[i])
        mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, .007,
                             _GUIDE_ORIGIN, _GUIDE_ORIGIN + direction * .16)
        label = scene.geoms[first + 3 + i]
        # MuJoCo's arrow primitive renders its tip halfway along this
        # connector, so place the label there rather than at the connector end.
        mujoco.mjv_initGeom(label, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.full(3, .001), _GUIDE_ORIGIN + direction * .08,
                            identity, _COLORS[i])
        label.label = _LABELS[i]
        tcp_arrow = scene.geoms[first + 6 + i]
        mujoco.mjv_initGeom(tcp_arrow, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                            np.zeros(3), identity, _COLORS[i])
    scene.ngeom += 9
    return first


def update_tcp_axes(scene, first, position):
    """Place small world-aligned arrows at the current TCP position."""
    position = np.asarray(position, dtype=float)
    for i, direction in enumerate(_DIRECTIONS):
        mujoco.mjv_connector(scene.geoms[first + 6 + i], mujoco.mjtGeom.mjGEOM_ARROW,
                             .004, position, position + direction * .085)

"""World-frame controller settings used by the native MuJoCo window."""
from .pose_control import PoseController


def make_native_controller(spec):
    """Build the same low-speed constrained pose controller used in MuJoCo."""
    return PoseController(dict(spec, mode='constrained_dls', posture_gain=0.,
                               cartesian_speed=.05, angular_speed=.3,
                               length_scale=.12, max_position_lead=.02,
                               max_orientation_lead=.08))

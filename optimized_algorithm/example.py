"""Minimal adapter example: supply a measured pose and 6-by-n Jacobian."""
import numpy as np

from optimized_algorithm import PoseController

spec = {
    "lower": [-2.0] * 6, "upper": [2.0] * 6,
    "velocity_limit": [1.0] * 6, "acceleration_limit": [3.0] * 6,
    "home": [0.0] * 6, "mode": "constrained_dls",
}
controller = PoseController(spec)
result = controller.step(
    q=np.zeros(6), qvel=np.zeros(6), tcp=np.zeros(3), rotation=np.eye(3),
    jacobian=np.eye(6), translation=[0.5, 0, 0], rotation_input=[0, 0, 0],
    enabled=True, dt=0.01,
)
print("joint reference:", result["q_ref"])
print("active constraints:", result["constraints"])

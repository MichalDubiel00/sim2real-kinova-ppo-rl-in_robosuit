"""
Registers the Kinova Gen3 6DOF in robosuite at runtime (the stock "Kinova3" is 7DOF).

MJCF + meshes: my_assets/robots/kinova3_6dof/, adapted from
https://github.com/goldengeko/robosuite_training_examples (Kortex inertias/torques).
Changes: RethinkMount base (Spot simulated as a rigid mount at KINOVA_BASE_Z),
Robotiq85 gripper (as on the lab Gen3), default_kinova3 controller.
Import before suite.make(robots="Kinova3_6DOF"); envs/__init__.py does it.
"""
import os

import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.robots import register_robot_class

_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "my_assets", "robots", "kinova3_6dof",
)


@register_robot_class("FixedBaseRobot")
class Kinova3_6DOF(ManipulatorModel):
    """Kinova Gen3 6DOF (lab variant)."""

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(os.path.join(_ASSETS_DIR, "robot.xml"), idn=idn)

    @property
    def default_base(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return {"right": "Robotiq85Gripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_kinova3"}

    @property
    def init_qpos(self):
        # Equivalent of the stock 7DOF folded pose (eef ~[0.43, 0, 1.61]). Pitch joints
        # are 2/3/5 here (2/4/6 in 7DOF); the goldengeko init_qpos was not remapped.
        return np.array([0.000, 0.650, -1.890, 0.000, -0.600, -np.pi / 2])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.5 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"

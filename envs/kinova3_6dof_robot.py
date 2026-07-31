"""
kinova3_6dof_robot.py
=====================
Rejestracja robota Kinova Gen3 6DOF w robosuite (runtime, bez kopiowania
do site-packages). Robot na uczelni to Gen3 w wariancie 6DOF; stockowy
"Kinova3" w robosuite to wariant 7DOF (inna kinematyka, o jeden staw
mniej w obserwacjach proprio).

Model MJCF + meshe: my_assets/robots/kinova3_6dof/ — zaadaptowane z
https://github.com/goldengeko/robosuite_training_examples (inercje i limity
momentow zgodne z oficjalnym opisem Kortex: duze aktuatory 39 Nm, male 9 Nm).

Zmiany wzgledem klasy z tego repo (parytet ze srodowiskiem uzywanym
dotad z 7DOF "Kinova3"):
  - default_base: RethinkMount zamiast SpotFloating (baze Spota symulujemy
    sztywnym montazem na KINOVA_BASE_Z, jak dotychczas),
  - default_gripper: Robotiq85Gripper zamiast Robotiq140 (taki jest na
    uczelnianym Gen3 i taki byl trenowany),
  - default_controller_config: default_kinova3 (OSC_POSE; skalarne gainy,
    dzialaja dla 6 stawow) — i tak zawsze podajemy jawnie composite BASIC.

Wystarczy zaimportowac ten modul PRZED suite.make(robots="Kinova3_6DOF") —
metaklasa RobotModelMeta rejestruje klase po nazwie, a dekorator
register_robot_class dopisuje mapowanie na FixedBaseRobot. Import jest
w envs/__init__.py, wiec kazdy skrypt importujacy envs (oraz workery
SubprocVecEnv) ma robota zarejestrowanego automatycznie.
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
    """Kinova Gen3 6DOF (wariant uczelniany; stockowy Kinova3 = 7DOF)."""

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
        # Odpowiednik zlozonej pozy stockowego 7DOF (eef ~[0.43, 0, 1.61]).
        # UWAGA: w 6DOF stawy pochylu to 2/3/5 (w 7DOF: 2/4/6) — init_qpos
        # z repo goldengeko byl skopiowany z 7DOF bez remapu i dawal ramie
        # wyprostowane w gore (eef z=2.67, poza zasiegiem tablicy).
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

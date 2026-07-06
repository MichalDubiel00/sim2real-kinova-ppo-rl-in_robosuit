"""
ButtonPress-v0
==============
Kinova Gen3 musi wcisnac przycisk Emergency Stop z odpowiednia sila.

Cel:
  - wcisnac przycisk > 12 mm (button_slide joint)
  - sila kontaktu w oknie [10 N, 25 N]
"""

import numpy as np
from collections import OrderedDict
import xml.etree.ElementTree as ET

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from objects.button_object import ButtonObject

_ROBOSUITE_PKG_PATH = os.path.split(robosuite.__file__)[0]


class ButtonPressEnv(ManipulationEnv):
    """
    Srodowisko wciskania przycisku Emergency Stop.

    Args:
        robots: "Kinova3"
        button_pos_noise: (float) +/- szum XY pozycji przycisku [m]
        force_min / force_max: okno akceptowanej sily [N]
    """

    BUTTON_ACTIVATION_M = 0.012  # przycisk aktywny przy >12 mm wcisniecia

    def __init__(
        self,
        robots="Kinova3",
        env_configuration="default",
        controller_configs=None,
        gripper_types="Robotiq85Gripper",
        initialization_noise=None,
        use_camera_obs=False,
        use_object_obs=True,
        reward_scale=1.0,
        horizon=500,
        button_pos_noise=0.10,
        force_min=10.0,
        force_max=25.0,
        **kwargs,
    ):
        self.button_pos_noise = button_pos_noise
        self.force_min = force_min
        self.force_max = force_max
        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale

        self._button_activated = False

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            horizon=horizon,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # XML processing
    # ------------------------------------------------------------------

    def edit_model_xml(self, xml_str):
        """Chroni wlasne assety przed przepisaniem sciezki przez robosuite."""
        path_split = _ROBOSUITE_PKG_PATH.split("/")

        tree = ET.fromstring(xml_str)
        asset = tree.find("asset")
        if asset is not None:
            for elem in asset.findall("mesh") + asset.findall("texture"):
                old_path = elem.get("file")
                if old_path is None or not old_path.startswith(_ROBOSUITE_PKG_PATH):
                    continue
                old_path_split = old_path.split("/")
                check_lst = [i for i, v in enumerate(old_path_split) if v == "robosuite"]
                if check_lst:
                    ind = max(check_lst)
                    new_path = "/".join(path_split + old_path_split[ind + 1:])
                    elem.set("file", new_path)

        return ET.tostring(tree, encoding="utf8").decode("utf8")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _load_model(self):
        super()._load_model()

        mujoco_arena = TableArena(
            table_full_size=(0.8, 0.8, 0.05),
            table_offset=(0, 0, 0.8),
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.button_obj = ButtonObject(name="estop")

        # Ustaw pozycje przycisku na blacie stolu (z szumem jesli chcemy)
        xy_noise = np.random.uniform(-self.button_pos_noise,
                                    self.button_pos_noise, size=2)
        btn_x = 0.25 + xy_noise[0]
        btn_y = xy_noise[1]
        btn_z = 0.8 + 0.05  # blat stolu
        self.button_obj.get_obj().set("pos", f"{btn_x:.4f} {btn_y:.4f} {btn_z:.4f}")

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.button_obj,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    # Stiffness sprężyny z button.xml [N/m]
    SPRING_STIFFNESS = 500.0

    def _setup_references(self):
        super()._setup_references()

        self.button_top_site_id = self.sim.model.site_name2id(
            "estop_button_top"
        )
        # Joint id i adres w qpos wektora
        jnt_id = self.sim.model.joint_name2id("estop_button_slide")
        self.button_jnt_qposadr = self.sim.model.jnt_qposadr[jnt_id]
        self.button_jnt_dofadr  = self.sim.model.jnt_dofadr[jnt_id]

    def _reset_internal(self):
        super()._reset_internal()
        self._button_activated = False
        # Reset sprężyny do pozycji 0 (nie wcisniety)
        self.sim.data.qpos[self.button_jnt_qposadr] = 0.0
        self.sim.data.qvel[self.button_jnt_dofadr]  = 0.0

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def _setup_observables(self):
        observables = super()._setup_observables()

        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality="object")
        def button_pos_relative(obs_cache):
            eef_pos = obs_cache.get(f"{pf}eef_pos", np.zeros(3))
            btn_top = self.sim.data.site_xpos[self.button_top_site_id]
            return btn_top - eef_pos

        @sensor(modality="object")
        def distance_to_button(obs_cache):
            eef_pos = obs_cache.get(f"{pf}eef_pos", np.zeros(3))
            btn_top = self.sim.data.site_xpos[self.button_top_site_id]
            return np.array([np.linalg.norm(btn_top - eef_pos)])

        @sensor(modality="object")
        def button_depression(obs_cache):
            return np.array([self.sim.data.qpos[self.button_jnt_qposadr]])

        @sensor(modality="object")
        def contact_force(obs_cache):
            # Sila sprężyny: F = k * x  (prawo Hooke'a), min 0 N
            x = max(0.0, float(self.sim.data.qpos[self.button_jnt_qposadr]))
            return np.array([self.SPRING_STIFFNESS * x])

        @sensor(modality="object")
        def eef_axis_vs_button_axis(obs_cache):
            eef_quat = obs_cache.get(f"{pf}eef_quat", np.array([0, 0, 0, 1]))
            eef_mat = T.quat2mat(eef_quat)
            eef_z = eef_mat[:, 2]
            # Button press axis is world -Z
            btn_axis = np.array([0, 0, -1.0])
            cos_a = np.clip(np.dot(eef_z, btn_axis), -1.0, 1.0)
            return np.array([np.arccos(np.abs(cos_a))])

        sensors = [button_pos_relative, distance_to_button,
                   button_depression, contact_force, eef_axis_vs_button_axis]
        names = ["button_pos_relative", "distance_to_button",
                 "button_depression", "contact_force", "eef_axis_vs_button_axis"]

        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def reward(self, action=None):
        obs = self._get_observations()

        dist = float(np.squeeze(obs["distance_to_button"]))
        axis_err = float(np.squeeze(obs["eef_axis_vs_button_axis"]))
        force = float(np.squeeze(obs["contact_force"]))
        depression = float(np.squeeze(obs["button_depression"]))

        r_approach = np.exp(-10.0 * dist)
        r_alignment = np.exp(-5.0 * axis_err)
        r_force = 1.0 if self.force_min <= force <= self.force_max else 0.0

        self._button_activated = depression >= self.BUTTON_ACTIVATION_M
        r_activation = 5.0 if self._button_activated else 0.0

        r_collision = -1.0 if self._check_wrong_collision() else 0.0

        reward = (1.0 * r_approach
                  + 1.0 * r_alignment
                  + 2.0 * r_force
                  + 5.0 * r_activation
                  + r_collision)

        if self.reward_scale is not None:
            reward *= self.reward_scale / 14.0

        return reward

    # ------------------------------------------------------------------
    # Success / termination
    # ------------------------------------------------------------------

    def _check_success(self):
        return self._button_activated

    def _check_wrong_collision(self):
        """True jesli robot uderza w cos innego niz przycisk."""
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            geom1 = self.sim.model.geom_id2name(contact.geom1) or ""
            geom2 = self.sim.model.geom_id2name(contact.geom2) or ""
            robot_geom = any(
                g.startswith(self.robots[0].robot_model.naming_prefix)
                for g in [geom1, geom2]
            )
            wrong = any("table" in g or "board" in g or "base" in g
                        for g in [geom1, geom2])
            if robot_geom and wrong:
                return True
        return False

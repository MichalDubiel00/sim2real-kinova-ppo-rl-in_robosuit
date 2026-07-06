"""
CylinderInspection-v0
=====================
SpotWithArm (Kinova Gen3 na Spot) podchodzi do boardu z 5 cylindrami.
Kinova musi ustawic wrist kamere tak, zeby patrzyla w otwar losowo wybranego cylindra.

Cel:
  - kat kamery vs os cylindra < 5 deg
  - odleglosc EEF od otworu < 5 cm
  - utrzymane przez 10 kolejnych krokow
"""

import numpy as np
import xml.etree.ElementTree as ET

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from objects.custom_objects import linearboardObject, N_SLOTS

_ROBOSUITE_PKG_PATH = os.path.split(robosuite.__file__)[0]

# Pozycja boardu wzgledem bazy Spota [m]:
#   X=1.0  — przed robotem
#   Z=0.5  — na wysokosci ~pol metra (np. na stoliku/podestcie)
BOARD_DEFAULT_POS = np.array([1.0, 0.0, 0.5])
BOARD_POS_NOISE   = np.array([0.10, 0.10, 0.0])   # szum XY przy resecie


class CylinderInspectionEnv(ManipulationEnv):
    """
    SpotWithArm zaglaada do cylindrow na linear_board.

    Args:
        robots           : "SpotWithArm"
        board_pos_noise  : (float) +/- szum XY pozycji boardu [m]
        board_yaw_noise  : (float) +/- szum yaw boardu [rad]
        success_hold_steps: ile krokow trzeba utrzymac sukces
        success_angle_thresh: [rad] prog kata kamery vs osi cylindra
        success_dist_thresh : [m]  prog odleglosci od otworu
    """

    def __init__(
        self,
        robots="SpotWithArm",
        env_configuration="default",
        controller_configs=None,
        gripper_types="Robotiq85Gripper",
        initialization_noise=None,
        use_camera_obs=False,
        use_object_obs=True,
        reward_scale=1.0,
        horizon=500,
        board_pos_noise=0.10,
        board_yaw_noise=np.deg2rad(15),
        success_hold_steps=10,
        success_angle_thresh=np.deg2rad(5),
        success_dist_thresh=0.05,
        **kwargs,
    ):
        self.board_pos_noise = board_pos_noise
        self.board_yaw_noise = board_yaw_noise
        self.success_hold_steps = success_hold_steps
        self.success_angle_thresh = success_angle_thresh
        self.success_dist_thresh = success_dist_thresh
        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale

        self._success_counter = 0
        self._target_slot = 0

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
    # XML processing — chronimy wlasne assety
    # ------------------------------------------------------------------

    def edit_model_xml(self, xml_str):
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
                    elem.set("file", "/".join(path_split + old_path_split[ind + 1:]))
        return ET.tostring(tree, encoding="utf8").decode("utf8")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _load_model(self):
        super()._load_model()

        # Pusta arena — tylko podloga, bez stolu
        mujoco_arena = EmptyArena()

        # Board z 5 cylindrami
        self.board_obj = linearboardObject(name="linearboard")

        # Losuj pozycje boardu
        xy_noise = np.random.uniform(-self.board_pos_noise, self.board_pos_noise, 2)
        yaw = np.random.uniform(-self.board_yaw_noise, self.board_yaw_noise)
        pos = BOARD_DEFAULT_POS + np.array([xy_noise[0], xy_noise[1], 0.0])
        quat = T.axisangle2quat(np.array([0, 0, yaw]))
        quat_str = " ".join(f"{v:.6f}" for v in quat)
        pos_str  = " ".join(f"{v:.6f}" for v in pos)
        self.board_obj.get_obj().set("pos",  pos_str)
        self.board_obj.get_obj().set("quat", quat_str)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.board_obj,
        )

    # ------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------

    def _setup_references(self):
        super()._setup_references()

        # Site IDs dla wszystkich 5 slotow
        self._opening_site_ids = [
            self.sim.model.site_name2id(f"linearboard_slot_{i}_opening")
            for i in range(N_SLOTS)
        ]
        self._axis_top_site_ids = [
            self.sim.model.site_name2id(f"linearboard_slot_{i}_axis_top")
            for i in range(N_SLOTS)
        ]

    def _reset_internal(self):
        super()._reset_internal()
        self._success_counter = 0
        # Wybierz losowy slot jako cel
        self._target_slot = np.random.randint(0, N_SLOTS)

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def _setup_observables(self):
        observables = super()._setup_observables()

        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality="object")
        def target_slot(obs_cache):
            return np.array([self._target_slot], dtype=np.float32)

        @sensor(modality="object")
        def target_pos_relative(obs_cache):
            eef_pos = obs_cache.get(f"{pf}eef_pos", np.zeros(3))
            opening = self.sim.data.site_xpos[self._opening_site_ids[self._target_slot]]
            return (opening - eef_pos).astype(np.float32)

        @sensor(modality="object")
        def target_axis_relative(obs_cache):
            # Os cylindra w ukladzie EEF
            opening  = self.sim.data.site_xpos[self._opening_site_ids[self._target_slot]]
            axis_top = self.sim.data.site_xpos[self._axis_top_site_ids[self._target_slot]]
            cyl_axis_world = axis_top - opening
            norm = np.linalg.norm(cyl_axis_world)
            if norm > 1e-6:
                cyl_axis_world /= norm
            eef_quat = obs_cache.get(f"{pf}eef_quat", np.array([0, 0, 0, 1]))
            eef_mat = T.quat2mat(eef_quat)
            return (eef_mat.T @ cyl_axis_world).astype(np.float32)

        @sensor(modality="object")
        def camera_axis_error(obs_cache):
            # Kat miedzy osia kamery (Z EEF) a osia cylindra [rad]
            eef_quat = obs_cache.get(f"{pf}eef_quat", np.array([0, 0, 0, 1]))
            eef_mat  = T.quat2mat(eef_quat)
            cam_axis = eef_mat[:, 2]
            opening  = self.sim.data.site_xpos[self._opening_site_ids[self._target_slot]]
            axis_top = self.sim.data.site_xpos[self._axis_top_site_ids[self._target_slot]]
            cyl_axis = axis_top - opening
            norm = np.linalg.norm(cyl_axis)
            if norm > 1e-6:
                cyl_axis /= norm
            cos_a = np.clip(np.dot(cam_axis, cyl_axis), -1.0, 1.0)
            return np.array([np.arccos(np.abs(cos_a))], dtype=np.float32)

        @sensor(modality="object")
        def distance_to_opening(obs_cache):
            eef_pos = obs_cache.get(f"{pf}eef_pos", np.zeros(3))
            opening = self.sim.data.site_xpos[self._opening_site_ids[self._target_slot]]
            return np.array([np.linalg.norm(opening - eef_pos)], dtype=np.float32)

        sensors = [target_slot, target_pos_relative, target_axis_relative,
                   camera_axis_error, distance_to_opening]
        names   = ["target_slot", "target_pos_relative", "target_axis_relative",
                   "camera_axis_error", "distance_to_opening"]

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

        angle_err = float(np.squeeze(obs["camera_axis_error"]))
        dist      = float(np.squeeze(obs["distance_to_opening"]))

        r_alignment = np.exp(-5.0  * angle_err)
        r_distance  = np.exp(-10.0 * dist)
        r_collision = -1.0 if self._check_robot_collision() else 0.0

        if angle_err < self.success_angle_thresh and dist < self.success_dist_thresh:
            self._success_counter += 1
        else:
            self._success_counter = 0

        r_stable = 1.0 if self._success_counter >= self.success_hold_steps else 0.0

        reward = (2.0 * r_alignment
                  + 1.0 * r_distance
                  + 5.0 * r_stable
                  + r_collision)

        if self.reward_scale is not None:
            reward *= self.reward_scale / 9.0

        return reward

    # ------------------------------------------------------------------
    # Success / termination
    # ------------------------------------------------------------------

    def _check_success(self):
        return self._success_counter >= self.success_hold_steps

    def _check_robot_collision(self):
        pf = self.robots[0].robot_model.naming_prefix
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            g1 = self.sim.model.geom_id2name(contact.geom1) or ""
            g2 = self.sim.model.geom_id2name(contact.geom2) or ""
            robot_touch = any(g.startswith(pf) for g in [g1, g2])
            bad = any("floor" in g or "board_collision" in g for g in [g1, g2])
            if robot_touch and bad:
                return True
        return False

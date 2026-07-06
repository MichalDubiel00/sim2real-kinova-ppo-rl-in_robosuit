"""
ButtonBoard-v0
==============
Kinova Gen3 (on Spot base) presses 5 Emergency Stop buttons
mounted in a cross (plus) pattern on the Omniboard wall panel.

Button layout (RoboCup omniboard): the cross-shaped block tapers toward
the robot; the centre button sits on the small front face pointing straight
at the robot, and the 4 arm buttons sit on the 45-degree bevelled faces of
the arms, each pointing diagonally outward (up/right/down/left + forward):

        [1]
    [4] [0] [2]      0 = front centre, then top / right / bottom / left
        [3]

Task: press all 5 buttons in sequence (order randomised when
random_order=True — the current target is exposed via target_onehot).

Sim2real observation model: the policy only sees quantities reproducible
on the real robot — target position/axis as estimated by an external
perception module (board pose from e.g. AprilTag), so both carry a
per-episode constant bias + per-step noise. Button depressions are NOT
observed (real board is not wired); press detection on hardware comes
from EEF force + push depth, and the task state machine (which button
is next) is the supervisor's job — here simulated by the env itself.

Coordinate frame:
  - Robot base at (0, 0, KINOVA_BASE_Z), facing +X
  - Wall face at x = WALL_X; omniboard big (back) face mounted on it,
    small front face at x = WALL_X - BOARD_DEPTH
  - Each button's spring joint compresses along the inward normal
    of the face it is mounted on
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
from objects.button_object import ButtonObject
from objects.custom_objects import OmniBoardObject

_ROBOSUITE_PKG_PATH = os.path.split(robosuite.__file__)[0]


class ButtonBoardEnv(ManipulationEnv):
    """
    5 Emergency Stop buttons in a cross pattern on the Omniboard, press in sequence.

    Args:
        robots:            "Kinova3"
        random_order:      if True, randomise press order each episode
        board_pos_noise:   +/- uniform noise [m] on board centre (Y, Z) per episode
        percep_pos_bias:   +/- uniform per-episode bias [m] on perceived target pos
                           (simulates board pose estimation error, e.g. AprilTag)
        percep_pos_noise:  gaussian per-step noise std [m] on perceived target pos
        percep_axis_noise: gaussian per-episode perturbation of the press axis
        reward_scale:      multiplier on total reward
    """

    N_BUTTONS = 5
    WALL_X = 0.50           # wall face; omniboard big (back) face sits here
    BOARD_DEPTH = 0.08      # omniboard depth (mesh: 80 mm)
    BOARD_Y = 0.0           # board lateral centre
    BOARD_Z = 1.20          # board centre height above floor
    KINOVA_BASE_Z = 0.65    # Spot body (~0.5m) + arm mount (~0.15m)

    # Per button: (offset of face centre from back-face centre, outward face
    # normal), both in world frame with the board mounted on the wall.
    # Bevel faces are 45 deg, their centres sit half the depth (40 mm) in
    # front of the wall and 120 mm from the cross centre.
    _S = 0.7071067811865476  # sqrt(2)/2
    BUTTON_FACES = [
        ((-0.08, 0.00, 0.00), (-1.0, 0.0, 0.0)),    # 0: front centre
        ((-0.04, 0.00, 0.12), (-_S, 0.0, _S)),      # 1: top bevel
        ((-0.04, 0.12, 0.00), (-_S, _S, 0.0)),      # 2: right bevel
        ((-0.04, 0.00, -0.12), (-_S, 0.0, -_S)),    # 3: bottom bevel
        ((-0.04, -0.12, 0.00), (-_S, -_S, 0.0)),    # 4: left bevel
    ]

    SPRING_STIFFNESS = 500.0  # N/m, matches button.xml
    ACTIVATION_DEPTH = 0.012  # 12 mm depression = button pressed

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
        random_order=False,
        board_pos_noise=0.05,
        percep_pos_bias=0.005,
        percep_pos_noise=0.002,
        percep_axis_noise=0.02,
        **kwargs,
    ):
        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale
        self.random_order = random_order
        self.board_pos_noise = board_pos_noise
        self.percep_pos_bias = percep_pos_bias
        self.percep_pos_noise = percep_pos_noise
        self.percep_axis_noise = percep_axis_noise

        self._current_idx = 0
        self._pressed = [False] * self.N_BUTTONS
        # Per-episode perception errors (redrawn in _reset_internal)
        self._bias_pos = np.zeros(3)
        self._bias_axis = np.zeros(3)

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
    # XML path fix (same workaround as other envs in this project)
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
                    new_path = "/".join(path_split + old_path_split[ind + 1:])
                    elem.set("file", new_path)
        return ET.tostring(tree, encoding="utf8").decode("utf8")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _btn_body_pos(self, i):
        """World position of button body origin (centre of its mounting face)."""
        dx, dy, dz = self.BUTTON_FACES[i][0]
        return (self.WALL_X + dx, self._board_y + dy, self._board_z + dz)

    @staticmethod
    def _face_quat(normal):
        """Quat (wxyz string) rotating button local +Z onto the face normal."""
        z = np.asarray(normal, dtype=float)
        z = z / np.linalg.norm(z)
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(up, z)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        q = T.mat2quat(np.column_stack([x, y, z]))  # xyzw
        return f"{q[3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}"

    def _load_model(self):
        super()._load_model()

        # Per-episode domain randomisation: board centre shifts in Y/Z
        yz_noise = np.random.uniform(-self.board_pos_noise,
                                     self.board_pos_noise, size=2)
        self._board_y = self.BOARD_Y + yz_noise[0]
        self._board_z = self.BOARD_Z + yz_noise[1]

        # Raise Kinova to Spot height
        self.robots[0].robot_model.set_base_xpos([0, 0, self.KINOVA_BASE_Z])

        mujoco_arena = EmptyArena()

        # Camera: side-front view — robot centre-left, wall+buttons right
        # pos=(-0.2, -1.5, 1.5) looking toward (0.3, 0, 1.2)
        # view direction (0.31, 0.93, -0.19), xyaxes computed from that
        cam = ET.SubElement(mujoco_arena.worldbody, "camera")
        cam.set("name", "board_view")
        cam.set("pos", "-0.2 -1.5 1.5")
        cam.set("xyaxes", "0.948 -0.316 0 0.059 0.176 0.981")

        # Visual wall panel behind the board (no collision)
        wall = ET.SubElement(mujoco_arena.worldbody, "geom")
        wall.set("name", "wall_panel")
        wall.set("type", "box")
        # half-sizes: 10 cm deep, 60 cm wide, 50 cm tall; face flush with board back
        wall.set("size", "0.10 0.60 0.50")
        wall.set("pos", f"{self.WALL_X + 0.10:.4f} "
                        f"{self._board_y:.4f} {self._board_z:.4f}")
        wall.set("rgba", "0.55 0.55 0.55 1.0")
        wall.set("contype", "0")
        wall.set("conaffinity", "0")

        # Omniboard: big (back) face on the wall at WALL_X, block tapers toward
        # the robot. Body local +Z (depth axis) must map to world -X:
        # rotate -90 deg around Y.
        self._board_obj = OmniBoardObject(name="omniboard")
        self._board_obj.get_obj().set(
            "pos", f"{self.WALL_X:.4f} {self._board_y:.4f} {self._board_z:.4f}")
        self._board_obj.get_obj().set("quat", "0.7071 0 -0.7071 0")

        # 5 buttons: centre on the front face, 4 on the 45-deg bevels
        self._button_objs = []
        for i in range(self.N_BUTTONS):
            btn = ButtonObject(name=f"btn{i}")
            bx, by, bz = self._btn_body_pos(i)
            btn.get_obj().set("pos", f"{bx:.4f} {by:.4f} {bz:.4f}")
            btn.get_obj().set("quat", self._face_quat(self.BUTTON_FACES[i][1]))
            self._button_objs.append(btn)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self._board_obj] + self._button_objs,
        )

    # ------------------------------------------------------------------
    # References (IDs into sim arrays, set up after model compile)
    # ------------------------------------------------------------------

    def _setup_references(self):
        super()._setup_references()

        self._btn_top_ids = [
            self.sim.model.site_name2id(f"btn{i}_button_top")
            for i in range(self.N_BUTTONS)
        ]

        jnt_ids = [
            self.sim.model.joint_name2id(f"btn{i}_button_slide")
            for i in range(self.N_BUTTONS)
        ]
        self._btn_qposadr = [self.sim.model.jnt_qposadr[j] for j in jnt_ids]
        self._btn_dofadr  = [self.sim.model.jnt_dofadr[j]  for j in jnt_ids]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_internal(self):
        super()._reset_internal()

        for i in range(self.N_BUTTONS):
            self.sim.data.qpos[self._btn_qposadr[i]] = 0.0
            self.sim.data.qvel[self._btn_dofadr[i]]  = 0.0

        self._current_idx = 0
        self._pressed = [False] * self.N_BUTTONS

        if self.random_order:
            self._order = np.random.permutation(self.N_BUTTONS).tolist()
        else:
            self._order = list(range(self.N_BUTTONS))

        # Per-episode perception error: constant board-pose estimation bias
        self._bias_pos = np.random.uniform(
            -self.percep_pos_bias, self.percep_pos_bias, size=3)
        self._bias_axis = np.random.normal(0.0, self.percep_axis_noise, size=3)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _target_btn(self):
        """Index of the button the robot should press right now."""
        if self._current_idx >= self.N_BUTTONS:
            return self.N_BUTTONS - 1  # all done, avoid out-of-bounds
        return self._order[self._current_idx]

    def _depression(self, i):
        return float(self.sim.data.qpos[self._btn_qposadr[i]])

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def _setup_observables(self):
        observables = super()._setup_observables()
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality="object")
        def target_relative_pos(obs_cache):
            """EEF->target vector as the PERCEPTION would see it:
            true vector + per-episode board-pose bias + per-step noise."""
            eef = obs_cache.get(f"{pf}eef_pos", np.zeros(3))
            tgt = self.sim.data.site_xpos[self._btn_top_ids[self._target_btn()]]
            noise = np.random.normal(0.0, self.percep_pos_noise, size=3)
            return (tgt - eef) + self._bias_pos + noise

        @sensor(modality="object")
        def target_distance(obs_cache):
            """Norm of the perceived (noisy) relative vector — consistent
            with target_relative_pos computed earlier this step."""
            rel = obs_cache.get("target_relative_pos")
            if rel is None:
                eef = obs_cache.get(f"{pf}eef_pos", np.zeros(3))
                tgt = self.sim.data.site_xpos[self._btn_top_ids[self._target_btn()]]
                rel = (tgt - eef) + self._bias_pos
            return np.array([np.linalg.norm(rel)])

        @sensor(modality="object")
        def target_onehot(obs_cache):
            """One-hot: which button is current target (5-dim)."""
            oh = np.zeros(self.N_BUTTONS, dtype=np.float32)
            oh[self._target_btn()] = 1.0
            return oh

        @sensor(modality="object")
        def target_press_axis(obs_cache):
            """Push direction of the current target as PERCEIVED
            (per-episode board orientation estimation error)."""
            n = np.asarray(self.BUTTON_FACES[self._target_btn()][1], dtype=np.float64)
            axis = -n + self._bias_axis
            return (axis / np.linalg.norm(axis)).astype(np.float32)

        @sensor(modality="object")
        def eef_align_with_target(obs_cache):
            """Angle [rad] between EEF +Z axis and the perceived press axis."""
            eef_quat = obs_cache.get(f"{pf}eef_quat", np.array([0, 0, 0, 1]))
            eef_z = T.quat2mat(eef_quat)[:, 2]
            axis = obs_cache.get("target_press_axis")
            if axis is None:
                axis = -np.asarray(self.BUTTON_FACES[self._target_btn()][1])
            cos_a = np.clip(np.dot(eef_z, axis), -1.0, 1.0)
            return np.array([np.arccos(np.abs(cos_a))])

        new_sensors = [
            target_relative_pos,
            target_distance,
            target_onehot,
            target_press_axis,
            eef_align_with_target,
        ]
        new_names = [
            "target_relative_pos",
            "target_distance",
            "target_onehot",
            "target_press_axis",
            "eef_align_with_target",
        ]

        for name, s in zip(new_names, new_sensors):
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
        tgt = self._target_btn()
        depress = self._depression(tgt)

        # Advance sequence when current button is activated
        press_bonus = 0.0
        if (
            self._current_idx < self.N_BUTTONS
            and depress >= self.ACTIVATION_DEPTH
            and not self._pressed[tgt]
        ):
            self._pressed[tgt] = True
            self._current_idx += 1
            press_bonus = 5.0

        # Reward uses TRUE state (sim-only signal, no perception noise)
        arm = self.robots[0].arms[0]
        eef_site = self.robots[0].eef_site_id[arm]
        eef_pos = self.sim.data.site_xpos[eef_site]
        tgt_pos = self.sim.data.site_xpos[self._btn_top_ids[self._target_btn()]]
        dist = float(np.linalg.norm(tgt_pos - eef_pos))

        eef_z = self.sim.data.site_xmat[eef_site].reshape(3, 3)[:, 2]
        axis = -np.asarray(self.BUTTON_FACES[self._target_btn()][1])
        cos_a = np.clip(np.dot(eef_z, axis), -1.0, 1.0)
        axis_err = float(np.arccos(np.abs(cos_a)))

        r_approach  = np.exp(-8.0 * dist)           # 0-1, approach target
        r_align     = np.exp(-3.0 * axis_err)        # 0-1, align EEF with push axis
        r_complete  = 20.0 if all(self._pressed) else 0.0
        r_collision = -0.5 if self._check_board_collision() else 0.0

        total = r_approach + 0.5 * r_align + press_bonus + r_complete + r_collision

        # Normalise: max theoretical reward per step ~ 1.5, press bonus 5, complete 20
        return total * self.reward_scale / (1.5 + 5.0 * self.N_BUTTONS + 20.0)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_success(self):
        return all(self._pressed)

    def _check_board_collision(self):
        """True if any robot geom touches the board plate (not the buttons)."""
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            geom1 = self.sim.model.geom_id2name(contact.geom1) or ""
            geom2 = self.sim.model.geom_id2name(contact.geom2) or ""
            robot_geom = any(g.startswith(("robot", "gripper"))
                             for g in (geom1, geom2))
            board_geom = any("board" in g for g in (geom1, geom2))
            if robot_geom and board_geom:
                return True
        return False

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

Press model (reward v2.2/v2.3): a press only counts when a gripper FINGER
geom is in contact with the button cap at the moment of activation —
mirroring the hardware supervisor, which detects presses from
force/kinematics at the TCP and would never register a press made with the
wrist or arm. ARM-link contact with any button is penalised like a board
collision (wrist-side pressing risks the Gen3 wrist camera and breaks the
TCP-based force/depth estimates). Gripper housing/knuckle contact is NOT
penalised (v2.3): it inevitably accompanies legitimate fingertip presses —
penalising it (v2.2) taught the policy to avoid buttons altogether
(0/100 successes). The gripper is also forced closed (action dim stays 7;
the gripper command is overridden) so the policy presses with a stiff,
well-defined fingertip contact.

Contact quality (reward v2.4, FAILED — see v2.5): v2.3 successfully unblocked
pressing but left NOTHING discouraging clumsy contact — buttons were hit at up
to 1 m/s (visible as gripper jaws splaying apart on hard/off-axis impact,
since the compliant Robotiq position controller loses to contact torque).
v2.4 added FINGER_MISUSE_COST for touching a non-target button and
CONTACT_SPEED_COST for fast contact with any button. Fine-tune result:
success dropped 85%->70%->43% with more training, while contact-quality
metrics stayed unchanged. ROOT CAUSE (found 2026-07-21): the target advances
on the same step a press succeeds, so the lingering finger contact with the
just-pressed cap during retreat counted as "wrong button" — measured ~93% of
all "wrong contact" steps were this artefact, meaning both penalties fired
right after every successful press and clawed back the press bonus. The
policy learned that pressing is unprofitable — same collapse mechanism as
v2.2. (The v2.3-era estimate "~25% of contact is bumping wrong buttons"
suffered the same measurement artefact; genuine bumping into unpressed
non-target buttons is ~2% of contact steps.)

Contact quality (reward v2.5, ALSO REJECTED 2026-07-21): both penalties fire
only on contact with a button that is neither the current target nor already
pressed. Lingering post-press contact and the correct final press are
penalty-free; only genuinely bumping a not-yet-pressed wrong button
(optionally fast) is penalized. Result: best 83/100 (below the 90/100 v2.3
baseline), still degrading with training; genuine bumping is only ~2% of
contact steps so the penalty had nothing to work on. Conclusion after three
failures (v2.2/v2.4/v2.5): contact grace belongs in the supervisor's
SafetyLayer (TCP speed limit near the board), not in the reward. The v2.5
penalties remain available as OPT-IN constructor params (finger_misuse_cost,
contact_speed_cost); the DEFAULT configuration is reward v2.3. The finger
press gate can be disabled (finger_press_gate=False) for curriculum stage 1
when training from scratch (e.g. for the 6DOF robot — see
envs/kinova3_6dof_robot.py).

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
        force_gripper_closed: if True, override the gripper action to
                           "close" every step (action_dim stays 7)
        finger_press_gate: if True (v2.2+), a press only counts when a
                           gripper finger touches the cap; False = any
                           contact that depresses the button counts
                           (curriculum stage 1 — the gate strangles
                           exploration when training from scratch)
        finger_misuse_cost: penalty per step of finger contact with a
                           non-target, not-yet-pressed button (v2.5;
                           default 0.0 = v2.3 semantics, v2.5 REJECTED)
        contact_speed_cost: penalty coef for EEF speed above
                           CONTACT_SPEED_THRESH during such contact
                           (v2.5; default 0.0 = v2.3 semantics)
        dense_align_coef:  weight of the axis-alignment dense term
                           (v2.3: 0.05; v2.4/v2.5 used 0.15)
        collision_cost:    penalty per step of arm/board contact
                           (v2: 0.5 — negligible; v2.1+: 5.0). For
                           FROM-SCRATCH training use the v2 value: with
                           5.0 a single collision step (-5/225) outweighs
                           the whole approach shaping (~10*0.7/225), so
                           exploration learns to avoid the board and
                           never discovers pressing (6DOF s1 failure,
                           2026-07-30)
        action_rate_cost:  penalty coef on ||a_t - a_{t-1}||^2 (v2.1+:
                           0.02; v2 scratch: 0.0)
        activation_depth:  cap depression [m] that counts as a press
                           (default 0.012 = historical 7DOF setting).
                           12 mm of the 15 mm travel is at the physical
                           margin of the Robotiq85+cap geometry: an
                           aligned centered pinch bottoms out on the
                           base ring at ~0.5 mm, a deliberate aligned
                           single-pad press reaches ~7 mm, the learned
                           oblique fingertip technique ~11 mm (6DOF,
                           2026-07-31). The real button's activation
                           depth is unknown and press detection on
                           hardware is electrical (supervisor) — the
                           sim threshold is a task-design parameter,
                           not ground truth. 6DOF uses 0.010.
        base_x_offset:     shifts the robot base toward the wall [m]
                           (default 0.0 = historical placement, base
                           axis 0.50 m from the wall face). Sim2real
                           knob: where Spot parks relative to the board.
                           The 6DOF needs to stand closer — at the
                           historical standoff its press poses require
                           folding into self-collision (base/pedestal)
                           or joint limits (docs 6.8).
        depress_shaping_k: potential-based shaping on the depression of
                           the CURRENT target: k * (depr_t - depr_{t-1}).
                           Pays for pushing the cap in, pays back on
                           release, zero for holding — same mechanism as
                           the distance shaping, on the press axis.
                           Discovery aid for 6DOF from-scratch training
                           (approach learned by 2.5M steps, dents up to
                           ~5 mm, but a full 12 mm press never sampled —
                           the 6DOF's tighter dexterous workspace makes
                           aligned pushes rare). Default 0.0 = v2.3.
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
    ACTIVATION_DEPTH = 0.012  # domyslne activation_depth (7DOF/v2.3);
    # 12 mm z 15 mm skoku jest przy geometrii Robotiq85+grzybek na granicy
    # fizycznej wykonalnosci — patrz parametr activation_depth w __init__

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
        force_gripper_closed=True,
        finger_press_gate=True,
        finger_misuse_cost=0.0,
        contact_speed_cost=0.0,
        dense_align_coef=0.05,
        collision_cost=5.0,
        action_rate_cost=0.02,
        depress_shaping_k=0.0,
        base_x_offset=0.0,
        activation_depth=None,
        **kwargs,
    ):
        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale
        self.random_order = random_order
        self.board_pos_noise = board_pos_noise
        self.percep_pos_bias = percep_pos_bias
        self.percep_pos_noise = percep_pos_noise
        self.percep_axis_noise = percep_axis_noise
        self.force_gripper_closed = force_gripper_closed
        self.finger_press_gate = finger_press_gate
        self.finger_misuse_cost = finger_misuse_cost
        self.contact_speed_cost = contact_speed_cost
        self.dense_align_coef = dense_align_coef
        self.collision_cost = collision_cost
        self.action_rate_cost = action_rate_cost
        self.depress_shaping_k = depress_shaping_k
        self.base_x_offset = base_x_offset
        self.activation_depth = (activation_depth if activation_depth is not None
                                 else self.ACTIVATION_DEPTH)

        self._current_idx = 0
        self._pressed = [False] * self.N_BUTTONS
        # Per-episode perception errors (redrawn in _reset_internal)
        self._bias_pos = np.zeros(3)
        self._bias_axis = np.zeros(3)
        # Potential-based shaping state (reward v2)
        self._prev_dist = None
        self._prev_target = None
        self._prev_action = None

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

        # Raise Kinova to Spot height; base_x_offset = Spot parked closer
        # to the wall (sim2real placement knob, see docstring)
        self.robots[0].robot_model.set_base_xpos(
            [self.base_x_offset, 0, self.KINOVA_BASE_Z])

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

        # Geom IDs for the v2.2 press gate / button-misuse penalty
        arm = self.robots[0].arms[0]
        gripper = self.robots[0].gripper[arm]
        finger_names = (gripper.important_geoms["left_finger"]
                        + gripper.important_geoms["right_finger"])
        self._finger_geom_ids = {self.sim.model.geom_name2id(n)
                                 for n in finger_names}
        self._cap_geom_ids = [
            self.sim.model.geom_name2id(f"btn{i}_cap_collision")
            for i in range(self.N_BUTTONS)
        ]
        self._btn_geom_ids = set(self._cap_geom_ids) | {
            self.sim.model.geom_name2id(f"btn{i}_base_collision")
            for i in range(self.N_BUTTONS)
        }
        # v2.3: misuse = only ARM links (robot0_*) on a button; gripper
        # housing/knuckles must stay free to touch during fingertip presses
        self._arm_geom_ids = {
            gid for gid in range(self.sim.model.ngeom)
            if (self.sim.model.geom_id2name(gid) or "").startswith("robot")
        }
        # v2.4: geom -> button index, to tell WHICH button a finger touches
        self._geom_to_btn = {}
        for i in range(self.N_BUTTONS):
            self._geom_to_btn[self._cap_geom_ids[i]] = i
            self._geom_to_btn[self.sim.model.geom_name2id(
                f"btn{i}_base_collision")] = i
        eef_site_id = self.robots[0].eef_site_id[arm]
        self._eef_site_name = self.sim.model.site_id2name(eef_site_id)

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

        # Reset shaping state (reward v2)
        self._prev_dist = None
        self._prev_target = None
        self._prev_action = None
        self._prev_depress = None
        self._prev_depress_target = None

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
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        if self.force_gripper_closed:
            action = np.asarray(action, dtype=np.float64).copy()
            action[-1] = 1.0  # Robotiq85: 1 = close
        return super().step(action)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    # Reward — wagi stale. Domyslna konfiguracja parametrow konstruktora
    # (finger_misuse_cost=0, contact_speed_cost=0, dense_align_coef=0.05,
    # finger_press_gate=True) = reward v2.3 (ostateczny; v2.5 ODRZUCONY,
    # jego kary sa dostepne jako opt-in przez parametry konstruktora).
    SHAPING_K = 10.0        # nagroda za kazdy metr zblizenia do celu
    PRESS_BONUS = 25.0      # za wcisniecie biezacego celu
    COMPLETE_BONUS = 100.0  # za komplet 5 przyciskow
    DENSE_COEF = 0.05       # sladowy skladnik odleglosci (gradient na starcie nauki)
    ACTION_COST = 0.01      # kara za wielkosc akcji
    ACTION_RATE_COST = 0.02  # domyslne action_rate_cost (v2.1+; v2 scratch: 0)
    COLLISION_COST = 5.0    # domyslne collision_cost (v2.1+; v2 scratch: 0.5)
    CONTACT_SPEED_THRESH = 0.15  # m/s; prog dla contact_speed_cost (v2.5)
    REWARD_NORM = 225.0     # ~max epizodu: 5*25 + 100

    def reward(self, action=None):
        tgt = self._target_btn()
        depress = self._depression(tgt)

        # Potential-based shaping on the target's depression (see docstring).
        # Uses the PRE-advance target; after a successful press the target
        # switches, the guard skips one step, so the released (pressed)
        # button never pays the shaping back — same guard as dist shaping.
        r_depress = 0.0
        if (self.depress_shaping_k
                and self._prev_depress_target == tgt
                and self._prev_depress is not None):
            r_depress = self.depress_shaping_k * (depress - self._prev_depress)
        self._prev_depress_target = tgt
        self._prev_depress = depress

        # Advance sequence when current button is activated. v2.2 gate:
        # the press must come from a gripper finger on the cap — the
        # hardware supervisor detects presses at the TCP and would never
        # register one made with the wrist or arm. The gate can be
        # disabled (finger_press_gate=False) for curriculum stage 1:
        # from-scratch training with the gate collapses to "do nothing".
        press_bonus = 0.0
        if (
            self._current_idx < self.N_BUTTONS
            and depress >= self.activation_depth
            and not self._pressed[tgt]
            and (not self.finger_press_gate or self._finger_on_cap(tgt))
        ):
            self._pressed[tgt] = True
            self._current_idx += 1
            press_bonus = self.PRESS_BONUS

        # Reward uses TRUE state (sim-only signal, no perception noise)
        arm = self.robots[0].arms[0]
        eef_site = self.robots[0].eef_site_id[arm]
        eef_pos = self.sim.data.site_xpos[eef_site]
        tgt = self._target_btn()  # may have advanced above
        tgt_pos = self.sim.data.site_xpos[self._btn_top_ids[tgt]]
        dist = float(np.linalg.norm(tgt_pos - eef_pos))

        eef_z = self.sim.data.site_xmat[eef_site].reshape(3, 3)[:, 2]
        axis = -np.asarray(self.BUTTON_FACES[tgt][1])
        cos_a = np.clip(np.dot(eef_z, axis), -1.0, 1.0)
        axis_err = float(np.arccos(np.abs(cos_a)))

        # Potential-based shaping: pays only for PROGRESS toward the target;
        # hovering earns 0, moving away pays back. No term on target switch
        # (a switched target must not look like a shaping penalty).
        if self._prev_target == tgt and self._prev_dist is not None:
            r_shape = self.SHAPING_K * (self._prev_dist - dist)
        else:
            r_shape = 0.0
        self._prev_target = tgt
        self._prev_dist = dist

        r_dense = (self.DENSE_COEF * np.exp(-8.0 * dist)
                   + self.dense_align_coef * np.exp(-3.0 * axis_err))
        r_complete = self.COMPLETE_BONUS if all(self._pressed) else 0.0
        r_collision = (-self.collision_cost
                       if (self._check_board_collision()
                           or self._check_button_misuse()) else 0.0)

        # v2.5 (fixes v2.4): penalize clumsy gripper contact — bumping a
        # wrong button, and doing so fast. Both penalties fire ONLY on
        # contact with a button that is neither the current target NOR
        # already pressed. The pressed-exclusion is critical: the target
        # advances on the very step a press succeeds, while the finger
        # keeps touching the just-pressed cap for several retreat steps —
        # in v2.4 those steps counted as "wrong contact" and (with fast
        # retreat) triggered the speed penalty too, so every successful
        # press was immediately punished (~93% of all "wrong contact"
        # steps were this artefact). Target contact stays penalty-free:
        # taxing correct presses collapsed the policy twice (v2.2, v2.4).
        r_finger_misuse = 0.0
        r_contact_speed = 0.0
        if self.finger_misuse_cost or self.contact_speed_cost:
            touched = self._finger_touched_buttons()
            wrong_touched = any(i != tgt and not self._pressed[i]
                                for i in touched)
            if wrong_touched:
                r_finger_misuse = -self.finger_misuse_cost
                if self.contact_speed_cost:
                    speed = float(np.linalg.norm(
                        self.sim.data.get_site_xvelp(self._eef_site_name)))
                    if speed > self.CONTACT_SPEED_THRESH:
                        r_contact_speed = -self.contact_speed_cost * (
                            speed - self.CONTACT_SPEED_THRESH)

        r_action = 0.0
        if action is not None:
            action = np.asarray(action, dtype=np.float64)
            r_action = -self.ACTION_COST * float(np.sum(np.square(action)))
            if self._prev_action is not None:
                delta = action - self._prev_action
                r_action -= self.action_rate_cost * float(np.sum(np.square(delta)))
            self._prev_action = action.copy()

        total = (r_shape + r_dense + r_depress + press_bonus + r_complete
                 + r_collision + r_action + r_finger_misuse + r_contact_speed)

        return total * self.reward_scale / self.REWARD_NORM

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_success(self):
        return all(self._pressed)

    def _post_action(self, action):
        """Terminate the episode once all buttons are pressed (reward v2):
        the completion bonus is a terminal reward, no post-success loitering."""
        reward, done, info = super()._post_action(action)
        if self._check_success():
            done = True
        return reward, done, info

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

    def _finger_on_cap(self, i):
        """True if any gripper finger geom touches button i's cap."""
        cap_id = self._cap_geom_ids[i]
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            g1, g2 = contact.geom1, contact.geom2
            if (g1 == cap_id and g2 in self._finger_geom_ids) or \
               (g2 == cap_id and g1 in self._finger_geom_ids):
                return True
        return False

    def _check_button_misuse(self):
        """True if an ARM link geom (robot0_*) touches any button — pressing
        with the wrist/arm risks the Gen3 wrist camera on hardware and
        invalidates TCP-based force/depth estimates. Gripper geoms are
        exempt (v2.3): knuckle/housing contact inevitably accompanies
        legitimate fingertip presses, and penalising it (v2.2) taught the
        policy to avoid buttons altogether."""
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            g1, g2 = contact.geom1, contact.geom2
            for r, b in ((g1, g2), (g2, g1)):
                if b in self._btn_geom_ids and r in self._arm_geom_ids:
                    return True
        return False

    def _finger_touched_buttons(self):
        """Set of button indices touched by ANY gripper finger geom this
        step. The reward (v2.5) penalizes contact with buttons that are
        neither the current target nor already pressed — genuine bumping
        on the way to the target, not post-press lingering contact."""
        touched = set()
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            g1, g2 = contact.geom1, contact.geom2
            for a, b in ((g1, g2), (g2, g1)):
                if a in self._finger_geom_ids and b in self._geom_to_btn:
                    touched.add(self._geom_to_btn[b])
        return touched

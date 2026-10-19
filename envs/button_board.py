"""
ButtonBoard-v0: Kinova Gen3 (on a Spot base) presses 5 E-stop buttons on the
cross-shaped Omniboard, in a sequence chosen by the supervisor (target_onehot).

        [1]
    [4] [0] [2]      0 = front centre face, 1-4 on the 45-degree bevels
        [3]

Observations hold only what the real robot can reproduce: target position/axis
from perception (per-episode bias + per-step noise); button depression is not
observed. A press counts only with a FINGER on the cap (finger_press_gate);
arm-link contact with a button is penalised like a board collision.

Default reward = v2.3; the v2.4/v2.5 contact penalties failed (docs 6.6) and are
opt-in. Frame: arm base at (base_x_offset, 0, KINOVA_BASE_Z) facing +X, board back
face on the wall at x = WALL_X.
"""

import numpy as np
import mujoco
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

# Button layout on the UNROLLED board: (face centre offset from the back-face centre,
# outward face normal), world frame. Radius 107 mm measured on the MASKOR model (2026-09-01).
_S = 0.7071067811865476  # sqrt(2)/2
_BUTTON_FACES_FLAT = [
    ((-0.08, 0.00, 0.00), (-1.0, 0.0, 0.0)),    # 0: front centre
    ((-0.04, 0.00, 0.107), (-_S, 0.0, _S)),      # 1: top bevel
    ((-0.04, 0.107, 0.00), (-_S, _S, 0.0)),      # 2: right bevel
    ((-0.04, 0.00, -0.107), (-_S, 0.0, -_S)),    # 3: bottom bevel
    ((-0.04, -0.107, 0.00), (-_S, -_S, 0.0)),    # 4: left bevel
]


def _roll_x(vec, ang):
    """Rotate a vector about world X (the board normal) by ang [rad]."""
    c, s = np.cos(ang), np.sin(ang)
    x, y, z = vec
    return (x, y * c - z * s, y * s + z * c)


def _rolled_button_faces(roll_deg):
    """Button layout after rolling the board by roll_deg about its normal (45 deg =
    diagonal layout, as mounted in Rescue League). The board body gets the same roll."""
    a = np.radians(roll_deg)
    return [(_roll_x(off, a), _roll_x(nrm, a)) for off, nrm in _BUTTON_FACES_FLAT]


class ButtonBoardEnv(ManipulationEnv):
    """
    5 E-stop buttons on the Omniboard, pressed in sequence.

    Args (selection; history in docs 6.x):
        random_order:        randomise the press order each episode
        board_pos_noise:     +/- board centre noise (Y, Z) per episode [m]
        percep_pos_bias:     +/- per-episode bias on the perceived target [m] (ArUco-scale)
        percep_pos_noise:    per-step gaussian noise on the perceived target [m]
        percep_axis_noise:   per-episode perturbation of the press axis
        base_z_noise:        +/- arm base height noise per episode [m]
                             Noise is in observations only; reward uses ground truth.
        force_gripper_closed: override the gripper action to "close" (action dim stays 7)
        finger_press_gate:   press counts only with a finger on the cap; False for
                             curriculum stage 1 (the gate stalls exploration from scratch)
        finger_misuse_cost, contact_speed_cost: v2.5 penalties (rejected), default 0
        dense_align_coef:    axis-alignment dense term weight (v2.3: 0.05)
        collision_cost:      per-step arm/board contact penalty (v2.1+: 5.0; use 0.5
                             from scratch, or the policy learns to avoid the board)
        action_rate_cost:    penalty on ||a_t - a_{t-1}||^2
        activation_depth:    cap travel counting as a press (6DOF uses 0.010; 12 mm is
                             at the physical limit of the Robotiq85 + cap geometry)
        base_x_offset:       arm base shift toward the wall [m] (where Spot parks)
        depress_shaping_k:   potential shaping on the target's depression (discovery aid)
        depress_clip_buffer: cap on the depth fed to that shaping (docs 6.18)
    """

    N_BUTTONS = 5
    WALL_X = 0.50           # wall face; omniboard big (back) face sits here
    BOARD_DEPTH = 0.08      # omniboard depth (mesh: 80 mm)
    BOARD_Y = 0.0           # board lateral centre
    # Board centre height measured on the MASKOR model (2026-09-01): 2 cm above the arm base.
    BOARD_Z = 0.64
    # ABSOLUTE arm base height (Spot ~0.5 m + mount). set_base_xpos() takes the pedestal
    # BOTTOM, so _load_model subtracts bottom_offset (docs 6.15). 0.62 from Webots TF;
    # Spot height is adjustable, hence base_z_noise.
    KINOVA_BASE_Z = 0.62

    # Board roll about its normal; 45 deg = diagonal layout (Rescue League).
    BOARD_ROLL_DEG = 45.0
    BUTTON_FACES = _rolled_button_faces(BOARD_ROLL_DEG)

    SPRING_STIFFNESS = 500.0  # N/m, matches button.xml
    ACTIVATION_DEPTH = 0.012  # default activation_depth (7DOF/v2.3)
    # Example latching E-stop profile (docs 6.33): f_start/f_peak/x_peak from the user
    # (15 -> 35 N at 3-4 mm), x_stroke from a datasheet; f_after and snap_width are assumptions.
    ESTOP_PROFILE_EXAMPLE = dict(f_start=15.0, f_peak=35.0, x_peak=0.0035,
                                 f_after=5.0, snap_width=0.0003, x_stroke=0.005,
                                 latch=True)

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
        board_pos_noise=0.06,
        percep_pos_bias=0.015,
        percep_pos_noise=0.005,
        percep_axis_noise=0.05,
        base_z_noise=0.03,
        gain_noise_lin=None,
        gain_noise_rot=None,
        joint_vel_limits=None,
        joint_acc_limits=None,
        joint_acc_limit_ref="qvel",
        joint_acc_limit_mode="clip",
        joint_vel_limit_mode="clip",
        eef_twist_limits=None,
        init_qpos=None,
        spot_body=False,
        spot_front_cam=False,
        obs_acc_saturation=False,
        action_rate_limit=None,
        action_delay_steps=None,
        acc_demand_cost=0.0,
        acc_margin_bonus=0.0,
        joint_acc_limits_noise=None,
        approach_speed_cost=0.0,
        approach_brake_time=0.30,
        spot_collision_cost=0.0,
        button_stiffness=None,
        button_estop_profile=None,
        force_gripper_closed=True,
        finger_press_gate=True,
        finger_misuse_cost=0.0,
        contact_speed_cost=0.0,
        dense_align_coef=0.05,
        collision_cost=5.0,
        action_rate_cost=0.02,
        depress_shaping_k=0.0,
        depress_clip_buffer=0.001,
        # Spot stand-off: -0.10 gives zero start contacts for 6DOF and 7DOF (docs 6.15).
        base_x_offset=-0.10,
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
        self.base_z_noise = base_z_noise
        self.force_gripper_closed = force_gripper_closed
        self.finger_press_gate = finger_press_gate
        self.finger_misuse_cost = finger_misuse_cost
        self.contact_speed_cost = contact_speed_cost
        self.dense_align_coef = dense_align_coef
        self.collision_cost = collision_cost
        self.action_rate_cost = action_rate_cost
        self.depress_shaping_k = depress_shaping_k
        self.depress_clip_buffer = depress_clip_buffer
        self.base_x_offset = base_x_offset
        self.activation_depth = (activation_depth if activation_depth is not None
                                 else self.ACTIVATION_DEPTH)

        # Per-episode, per-axis gain randomisation (lo, hi) or None (docs 6.30).
        self.gain_noise_lin = gain_noise_lin
        self.gain_noise_rot = gain_noise_rot
        # Joint velocity limits [rad/s] (6 values) or None: simulator FIDELITY, not
        # randomisation. Enforced after every physics substep (hook on sim.step/step2):
        # v = clip(v), q = q_prev + v*dt. Pre-step saturation leaked up to 2.9x (docs 6.32).
        self.joint_vel_limits = (None if joint_vel_limits is None
                                 else np.asarray(joint_vel_limits, dtype=np.float64))
        # Joint acceleration limits [rad/s^2], off by default: the 1.0/10.0 values may be
        # MoveIt planner limits, not firmware (docs 6.31).
        self.joint_acc_limits = (None if joint_acc_limits is None
                                 else np.asarray(joint_acc_limits, dtype=np.float64))
        # Velocity limit mode (docs 6.37): "clip" per joint (changes direction),
        # "scale" all joints proportionally (direction kept; Gen3 twist mode).
        if joint_vel_limit_mode not in ("clip", "scale"):
            raise ValueError(f"joint_vel_limit_mode: {joint_vel_limit_mode}")
        self.joint_vel_limit_mode = joint_vel_limit_mode
        # Tool speed caps (lin [m/s], ang [rad/s]) or None; Gen3 twist: 0.5 / 0.8727,
        # proportional. Applied to realised joint velocity via damped least squares.
        self.eef_twist_limits = (None if eef_twist_limits is None
                                 else (float(eef_twist_limits[0]), float(eef_twist_limits[1])))
        self._limit_debug = False      # tests: store the last substep in _limit_last
        self._limit_last = None
        # Cap spring stiffness [N/m] or None (button.xml: 500). Real E-stops need 20-40 N
        # (docs 6.33). Start pose: None = robot default (gripper pointing down, the wrist
        # camera sees the floor); a 6-value list overrides it (docs 6.42).
        self.init_qpos = (None if init_qpos is None
                          else np.asarray(init_qpos, dtype=np.float64))
        # Spot body under the arm, measured from Webots Spot.proto (2026-10-01) incl. box
        # rotations; arm base frame: X -0.729..+0.266, Y +-0.139, Z -0.202..+0.002.
        # Off by default (changes the task vs runs 6.1-6.41).
        self.spot_body = bool(spot_body)
        # Front Spot camera, modelled as the Kinect sensor bar (a deliberate buffer; the lab
        # camera is smaller). Arm base frame: X +0.209..+0.273, Y +-0.140, Z -0.026..+0.012.
        self.spot_front_cam = bool(spot_front_cam)
        # Saturation signal in the observation (docs 6.47). Off by default: obs 120 -> 132.
        self.obs_acc_saturation = bool(obs_acc_saturation)
        # Command rate LIMIT (docs 6.47): |a_t - a_{t-1}| <= action_rate_limit per component,
        # before the controller. A constraint, not a penalty - penalties were paid off.
        # The bridge can apply the same limit.
        self.action_rate_limit = (None if action_rate_limit is None
                                  else float(action_rate_limit))
        # Loop dead time (docs 6.52): robosuite has none. Modelled as ACTION delay (same
        # loop effect as observation delay, obs size unchanged).
        #   int       fixed delay
        #   [lo, hi]  per-episode random delay
        # Take the range from MEASUREMENT (ros2_bridge/diag_latency.py), not a guess.
        if action_delay_steps is None:
            self.action_delay_steps = None
        elif isinstance(action_delay_steps, (list, tuple)):
            self.action_delay_steps = (int(action_delay_steps[0]), int(action_delay_steps[1]))
        else:
            self.action_delay_steps = int(action_delay_steps)
        self._delay_buf = None      # commands in flight
        self._delay_d = 0           # delay drawn for this episode
        self._prev_cmd = None
        # Variant A (docs 6.46): penalty on DEMANDED acceleration above the limit plus a
        # bonus for margin below it (the realised one is clipped, so always compliant).
        # The bonus gives a gradient on both sides of the threshold.
        self.acc_demand_cost = float(acc_demand_cost)
        self.acc_margin_bonus = float(acc_margin_bonus)
        self._acc_demand = None      # [rad/s^2] demand from the last control step
        # Acceleration limit reference (docs 6.48): "qvel" = measured velocity (eaten by
        # OSC ripple), "cmd" = own previous command (what the Kinova firmware limits).
        # Default "qvel" keeps the evaluation history valid.
        self.joint_acc_limit_ref = str(joint_acc_limit_ref)
        # Acceleration clipping (docs 6.50): "clip" per joint (changes direction), "scale"
        # one factor for the whole increment (direction kept, as Gen3 twist mode).
        self.joint_acc_limit_mode = str(joint_acc_limit_mode)
        self._v_cmd_prev = None      # reference for "cmd" mode

        # Variant B (docs 6.46): per-episode acceleration limit multiplier [lo, hi].
        self.joint_acc_limits_noise = (None if joint_acc_limits_noise is None
                                       else (float(joint_acc_limits_noise[0]),
                                             float(joint_acc_limits_noise[1])))
        self._acc_limits_base = None
        # Approach-speed penalty (docs 6.45): after the Gen3 acceleration limit, collisions
        # rose 0.4 -> 67 steps/episode because the clipped command could not brake in time.
        # Penalises approach speed above v_ok = distance / approach_brake_time, toward the
        # target only (lateral motion and retreat are free).
        self.approach_speed_cost = float(approach_speed_cost)
        self.approach_brake_time = float(approach_brake_time)
        # Spot-body contact penalty per step, before normalisation (225). Do NOT reuse 5.0:
        # hundreds of contact steps per episode would teach the policy to stand still (6.20).
        self.spot_collision_cost = float(spot_collision_cost)
        self.button_stiffness = button_stiffness
        # Non-linear latching E-stop model, or None (= linear spring). Force rises f_start ->
        # f_peak at x_peak, drops to f_after over snap_width, then constant; travel capped
        # at x_stroke; latch=True keeps the cap down until reset. The press counts at the
        # force drop unless activation_depth is given.
        self.button_estop_profile = (None if button_estop_profile is None
                                     else dict(button_estop_profile))
        if self.button_estop_profile is not None:
            _p = self.button_estop_profile
            _miss = {"f_start", "f_peak", "x_peak", "f_after", "snap_width", "x_stroke"} - set(_p)
            if _miss:
                raise ValueError(f"button_estop_profile: missing keys {sorted(_miss)}")
            if self.button_stiffness is not None:
                raise ValueError("button_stiffness and button_estop_profile are mutually exclusive")
            if activation_depth is None:
                self.activation_depth = _p["x_peak"] + _p["snap_width"]
            if self.activation_depth > _p["x_stroke"]:
                raise ValueError("activation_depth greater than x_stroke")
        self._btn_latched = [False] * self.N_BUTTONS
        self._gain = np.ones(6)
        self._raw_action = None

        self._base_z = self.KINOVA_BASE_Z
        self._current_idx = 0
        self._pressed = [False] * self.N_BUTTONS
        # Per-episode perception errors (redrawn in _reset_internal)
        self._bias_pos = np.zeros(3)
        self._bias_axis = np.zeros(3)
        # Potential-based shaping state (reward v2)
        self._prev_dist = None
        self._prev_target = None
        self._prev_action = None
        self._prev_cmd = None

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

        # Arm at Spot height; base_x_offset = Spot parked closer to the wall.
        # set_base_xpos() places the pedestal BOTTOM (bottom_offset is negative).
        # Base height is randomised: base pose was the only measured transfer failure
        # mode (docs 6.22), since obs contains absolute joint angles and EEF position.
        self._gain = np.ones(6)
        if self.gain_noise_lin is not None:
            lo, hi = self.gain_noise_lin
            self._gain[0:3] = np.random.uniform(lo, hi, 3)
        if self.gain_noise_rot is not None:
            lo, hi = self.gain_noise_rot
            self._gain[3:6] = np.random.uniform(lo, hi, 3)

        if self.joint_acc_limits_noise is not None and self.joint_acc_limits is not None:
            if self._acc_limits_base is None:
                self._acc_limits_base = np.array(self.joint_acc_limits, dtype=np.float64)
            lo, hi = self.joint_acc_limits_noise
            self.joint_acc_limits = self._acc_limits_base * float(np.random.uniform(lo, hi))

        self._base_z = self.KINOVA_BASE_Z + np.random.uniform(
            -self.base_z_noise, self.base_z_noise)
        mount_z = float(self.robots[0].robot_model.bottom_offset[2])
        self.robots[0].robot_model.set_base_xpos(
            [self.base_x_offset, 0, self._base_z + mount_z])
        if self.init_qpos is not None:
            # init_qpos is a property without a setter: set the instance attribute,
            # which robosuite applies in _reset_internal (survives hard_reset).
            self.robots[0].init_qpos = self.init_qpos.copy()

        mujoco_arena = EmptyArena()

        # Camera: side-front view — robot centre-left, wall+buttons right
        # pos=(-0.2, -1.5, 1.5) looking toward (0.3, 0, 1.2)
        # view direction (0.31, 0.93, -0.19), xyaxes computed from that
        cam = ET.SubElement(mujoco_arena.worldbody, "camera")
        cam.set("name", "board_view")
        # Framed for the docs 6.15 geometry (~4 cm low since 2026-09-01); recordings only.
        cam.set("pos", "-0.25 -0.75 0.95")
        cam.set("xyaxes", "0.7458 -0.6662 0 0.2190 0.2452 0.9445")

        # PHYSICAL wall behind the board (visual-only before 2026-08-05, letting the arm
        # reach through the board; docs 6.19). The flush board/wall pair is static, so
        # MuJoCo filters it.
        wall = ET.SubElement(mujoco_arena.worldbody, "geom")
        wall.set("name", "wall_panel")
        wall.set("type", "box")
        # half-sizes: 10 cm deep, 60 cm wide, 50 cm tall; face flush with board back
        wall.set("size", "0.10 0.60 0.50")
        wall.set("pos", f"{self.WALL_X + 0.10:.4f} "
                        f"{self._board_y:.4f} {self._board_z:.4f}")
        wall.set("rgba", "0.55 0.55 0.55 1.0")
        # group 1 = visual group; otherwise robosuite does not render the geom.
        wall.set("group", "1")
        wall.set("contype", "1")
        wall.set("conaffinity", "1")

        if self.spot_body:
            # One box covering body + front section (conservative).
            spot = ET.SubElement(mujoco_arena.worldbody, "geom")
            spot.set("name", "spot_body_collision")
            spot.set("type", "box")
            # half-sizes: (0.266+0.729)/2, 0.139, (0.002+0.202)/2
            spot.set("size", "0.4975 0.1390 0.1020")
            spot.set("pos", f"{self.base_x_offset - 0.2315:.4f} 0 "
                            f"{self._base_z - 0.1000:.4f}")
            spot.set("rgba", "0.85 0.45 0.10 1.0")   # orange: graphite blended with the pedestal
            spot.set("group", "1")   # visual group, otherwise not rendered
            spot.set("contype", "1")
            spot.set("conaffinity", "1")

        if self.spot_front_cam:
            cam_g = ET.SubElement(mujoco_arena.worldbody, "geom")
            cam_g.set("name", "spot_frontcam_collision")
            cam_g.set("type", "box")
            cam_g.set("size", "0.0320 0.1400 0.0190")
            cam_g.set("pos", f"{self.base_x_offset + 0.2410:.4f} 0 "
                             f"{self._base_z - 0.0070:.4f}")
            cam_g.set("rgba", "0.10 0.55 0.85 1.0")  # blue, to tell the bar from the body
            cam_g.set("group", "1")
            cam_g.set("contype", "1")
            cam_g.set("conaffinity", "1")

        # Omniboard: big (back) face on the wall at WALL_X, block tapers toward
        # the robot. Body local +Z (depth axis) must map to world -X:
        # rotate -90 deg around Y.
        self._board_obj = OmniBoardObject(name="omniboard")
        self._board_obj.get_obj().set(
            "pos", f"{self.WALL_X:.4f} {self._board_y:.4f} {self._board_z:.4f}")
        # The board body gets the same roll as BUTTON_FACES: R = Rx(roll) @ Ry(-90).
        a = np.radians(self.BOARD_ROLL_DEG)
        rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(a), -np.sin(a)],
                       [0.0, np.sin(a), np.cos(a)]])
        ry = np.array([[0.0, 0.0, -1.0],
                       [0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0]])
        bq = T.mat2quat(rx @ ry)  # xyzw
        self._board_obj.get_obj().set(
            "quat", f"{bq[3]:.6f} {bq[0]:.6f} {bq[1]:.6f} {bq[2]:.6f}")

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
            self.sim.data.qfrc_applied[self._btn_dofadr[i]] = 0.0
        self._btn_latched = [False] * self.N_BUTTONS
        self._v_cmd_prev = None
        if self.action_delay_steps is None:
            self._delay_d, self._delay_buf = 0, None
        else:
            d = self.action_delay_steps
            self._delay_d = (int(np.random.randint(d[0], d[1] + 1))
                             if isinstance(d, tuple) else int(d))
            # The queue starts with ZERO commands ("stand still"), as on hardware.
            self._delay_buf = [np.zeros(self.action_dim) for _ in range(self._delay_d)]

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

        @sensor(modality="object")
        def joint_acc_saturation(obs_cache):
            """How much the acceleration limit CLIPPED the command, per joint: 0 = within
            the envelope, 1.0 = demand twice the limit (capped). Makes the clipping
            observable (docs 6.47). Zero when the limit is off."""
            if self.joint_acc_limits is None or self._acc_demand is None:
                return np.zeros(6, dtype=np.float32)
            lim = np.asarray(self.joint_acc_limits, dtype=np.float64)
            nad = np.clip((self._acc_demand - lim) / lim, 0.0, 1.0)
            return nad.astype(np.float32)

        new_sensors = ([joint_acc_saturation] if self.obs_acc_saturation else []) + [
            target_relative_pos,
            target_distance,
            target_onehot,
            target_press_axis,
            eef_align_with_target,
        ]
        new_names = (["joint_acc_saturation"] if self.obs_acc_saturation else []) + [
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

    # ------------------------------------------------------------------
    # Joint drive limits (docs 6.31, 6.32)
    # ------------------------------------------------------------------

    def _initialize_sim(self, xml_string=None):
        super()._initialize_sim(xml_string)
        # hard_reset creates a NEW sim each reset: reinstall the hook every time.
        self._install_joint_limit_hook()
        self._apply_button_model()

    def _apply_button_model(self):
        # Applied after model compile on every init; getattr because the first init may run
        # inside super().__init__ before the attributes exist.
        k = getattr(self, "button_stiffness", None)
        prof = getattr(self, "button_estop_profile", None)
        if k is None and prof is None:
            return
        M = self.sim.model._model
        for i in range(self.N_BUTTONS):
            jid = self.sim.model.joint_name2id(f"btn{i}_button_slide")
            if prof is not None:
                M.jnt_stiffness[jid] = 0.0          # force comes from the profile, not the spring
                M.jnt_range[jid, 1] = float(prof["x_stroke"])
            else:
                M.jnt_stiffness[jid] = float(k)

    def _estop_force(self, x):
        """Force [N] opposing the press at depth x [m], from button_estop_profile."""
        p = self.button_estop_profile
        if x <= 0.0:
            return p["f_start"]
        if x < p["x_peak"]:
            return p["f_start"] + (p["f_peak"] - p["f_start"]) * x / p["x_peak"]
        x1 = p["x_peak"] + p["snap_width"]
        if x < x1:
            return p["f_peak"] + (p["f_after"] - p["f_peak"]) * (x - p["x_peak"]) / p["snap_width"]
        return p["f_after"]

    def _install_joint_limit_hook(self):
        # Hook both step1/step2 (lite_physics) and step. Call the CLASS implementation on
        # the current self.sim, never a bound self.sim method: the closure would form a
        # cycle that keeps old MjSims alive after hard resets (OOM, docs 6.32).
        cls = type(self.sim)
        for name in ("step", "step2"):
            fn = getattr(cls, name, None)
            if fn is not None:
                setattr(self.sim, name, self._limited_integration(fn))

    def _limited_integration(self, fn):
        def integrate(*args, **kwargs):
            d = self.sim.data
            prof = getattr(self, "button_estop_profile", None)
            btn_ok = prof is not None and getattr(self, "_btn_qposadr", None) is not None
            if btn_ok:
                if getattr(self, "_btn_latched", None) is None:
                    self._btn_latched = [False] * self.N_BUTTONS
                for i in range(self.N_BUTTONS):
                    da = self._btn_dofadr[i]
                    if self._btn_latched[i]:
                        d.qfrc_applied[da] = 0.0
                    else:
                        d.qfrc_applied[da] = -self._estop_force(float(d.qpos[self._btn_qposadr[i]]))
            lim_v, lim_a = self.joint_vel_limits, self.joint_acc_limits
            tw = getattr(self, "eef_twist_limits", None)
            robots = getattr(self, "robots", None)
            arm_ok = not ((lim_v is None and lim_a is None and tw is None) or not robots or
                          getattr(robots[0], "_ref_joint_vel_indexes", None) is None)
            if arm_ok:
                vidx = robots[0]._ref_joint_vel_indexes
                qidx = robots[0]._ref_joint_pos_indexes
                q_prev = np.array(d.qpos[qidx], dtype=np.float64)
                v_prev = np.array(d.qvel[vidx], dtype=np.float64)
            out = fn(self.sim, *args, **kwargs)
            if arm_ok:
                v = np.array(d.qvel[vidx], dtype=np.float64)
                v_in = v.copy()
                if tw is not None:
                    M = self.sim.model._model
                    site = robots[0].eef_site_id[robots[0].arms[0]]
                    jacp = np.zeros((3, M.nv)); jacr = np.zeros((3, M.nv))
                    mujoco.mj_jacSite(M, self.sim.data._data, jacp, jacr, site)
                    J = np.vstack([jacp[:, vidx], jacr[:, vidx]])
                    t = J @ v
                    nl, na = np.linalg.norm(t[:3]), np.linalg.norm(t[3:])
                    if nl > tw[0] or na > tw[1]:
                        t2 = np.concatenate([t[:3] * min(1.0, tw[0] / max(nl, 1e-12)),
                                             t[3:] * min(1.0, tw[1] / max(na, 1e-12))])
                        lam2 = 1e-6
                        v = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), t2)
                v_tw = v.copy()
                if lim_v is not None:
                    if getattr(self, "joint_vel_limit_mode", "clip") == "scale":
                        ratio = np.max(np.abs(v) / lim_v)
                        if ratio > 1.0:
                            v = v / ratio
                    else:
                        v = np.clip(v, -lim_v, lim_v)
                # Acceleration clip LAST: earlier, twist caps and the velocity limit changed v
                # afterwards (j1 hit 33.9 rad/s^2 vs 5.2). Clipping toward v_prev stays inside
                # the (convex) velocity envelope.
                if lim_a is not None:
                    # Reference: measured velocity or own previous command (joint_acc_limit_ref).
                    if self.joint_acc_limit_ref == "cmd":
                        if self._v_cmd_prev is None:
                            self._v_cmd_prev = v_prev.copy()
                        v_ref = self._v_cmd_prev
                    else:
                        v_ref = v_prev
                    # DEMAND before clipping, the only place it is visible
                    zad = np.abs(v - v_ref) / max(self.model_timestep, 1e-12)
                    self._acc_demand = (zad if self._acc_demand is None
                                        else np.maximum(self._acc_demand, zad))
                    dv = lim_a * self.model_timestep
                    if self.joint_acc_limit_mode == "scale":
                        # one factor for the whole increment, direction kept
                        d_req = v - v_ref
                        ratio = np.max(np.abs(d_req) / dv)
                        if ratio > 1.0:
                            v = v_ref + d_req / ratio
                    else:
                        v = np.clip(v, v_ref - dv, v_ref + dv)
                    self._v_cmd_prev = v.copy()
                if getattr(self, "_limit_debug", False):
                    self._limit_last = dict(v_in=v_in, v_tw=v_tw, v=v.copy(), q_prev=q_prev,
                                            J=(J if tw is not None else None))
                d.qvel[vidx] = v
                d.qpos[qidx] = q_prev + v * self.model_timestep
            if btn_ok:
                # Hard kinematic stops, like a latch: the soft MuJoCo joint limit yields
                # 8 mm under a 15 N preload on the 36 g cap (docs 6.34).
                xs = float(prof["x_stroke"])
                latch = prof.get("latch", True)
                for i in range(self.N_BUTTONS):
                    qa, da = self._btn_qposadr[i], self._btn_dofadr[i]
                    if latch and (self._btn_latched[i] or d.qpos[qa] >= xs - 1e-4):
                        self._btn_latched[i] = True
                        d.qpos[qa] = xs
                        d.qvel[da] = 0.0
                    elif d.qpos[qa] <= 0.0:
                        d.qpos[qa] = 0.0
                        d.qvel[da] = max(float(d.qvel[da]), 0.0)
                    elif d.qpos[qa] >= xs:
                        d.qpos[qa] = xs
                        d.qvel[da] = min(float(d.qvel[da]), 0.0)
            return out
        return integrate

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).copy()
        if self.force_gripper_closed:
            action[-1] = 1.0  # Robotiq85: 1 = close
        # Action penalties use the POLICY's output, before gain randomisation (docs 6.30).
        if self.action_rate_limit is not None:
            if self._prev_cmd is not None:
                d = self.action_rate_limit
                action[0:6] = np.clip(action[0:6], self._prev_cmd[0:6] - d,
                                      self._prev_cmd[0:6] + d)
            self._prev_cmd = action.copy()
        # ... but after the rate limit, which is the command that reaches the controller.
        self._raw_action = action.copy()
        # Delay applied AFTER `_raw_action`, so penalties refer to what the policy issued.
        if self._delay_buf is not None and self._delay_d > 0:
            self._delay_buf.append(action.copy())
            action = self._delay_buf.pop(0)
        if self.gain_noise_lin is not None or self.gain_noise_rot is not None:
            action[0:6] = np.clip(action[0:6] * self._gain, -1.0, 1.0)
        return super().step(action)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    # Fixed weights; default constructor args = reward v2.3.
    SHAPING_K = 10.0        # per metre of approach
    PRESS_BONUS = 25.0      # per press of the current target
    COMPLETE_BONUS = 100.0  # for all 5 buttons
    DENSE_COEF = 0.05       # small distance term (early gradient)
    ACTION_COST = 0.01      # action magnitude penalty
    ACTION_RATE_COST = 0.02  # default action_rate_cost (v2.1+)
    COLLISION_COST = 5.0    # default collision_cost (v2.1+)
    CONTACT_SPEED_THRESH = 0.15  # m/s; contact_speed_cost threshold (v2.5)
    REWARD_NORM = 225.0     # ~max episode return: 5*25 + 100

    def reward(self, action=None):
        if self._raw_action is not None:
            action = self._raw_action        # see step(): penalties on the policy's action
        tgt = self._target_btn()
        depress = self._depression(tgt)

        # Potential shaping on the target's depression, using the PRE-advance target.
        # The depth is capped at activation_depth + depress_clip_buffer: deeper travel
        # only buys contact force, which collision_cost penalises (docs 6.18). The capped
        # depth is still a state function, so the optimal policy is unchanged.
        depress_shaped = min(depress,
                             self.activation_depth + self.depress_clip_buffer)
        r_depress = 0.0
        if (self.depress_shaping_k
                and self._prev_depress_target == tgt
                and self._prev_depress is not None):
            r_depress = self.depress_shaping_k * (depress_shaped
                                                  - self._prev_depress)
        self._prev_depress_target = tgt
        self._prev_depress = depress_shaped

        # Advance the sequence on activation. With finger_press_gate the press must come
        # from a finger on the cap.
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

        # Potential shaping: pays only for progress; no term on target switch.
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
        # Spot body: separate coefficient (contacts are far more frequent than with the board).
        if self.spot_collision_cost and self._check_spot_collision():
            r_collision -= self.spot_collision_cost

        # v2.5 (opt-in): penalise touching a button that is neither the target nor already
        # pressed, and doing so fast. Excluding pressed buttons matters: in v2.4 the lingering
        # contact after every press was penalised (~93 % of 'wrong contact').
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

        r_acc = 0.0
        if (self.acc_demand_cost or self.acc_margin_bonus) and self._acc_demand is not None \
                and self.joint_acc_limits is not None:
            lim = self.joint_acc_limits
            # LINEAR and capped at 3x the limit: the quadratic version scored contact
            # impulses (max over substeps), giving -274 per episode.
            nad = np.clip((self._acc_demand - lim) / lim, 0.0, 3.0)
            zapas = np.maximum(lim - self._acc_demand, 0.0) / lim
            r_acc = (-self.acc_demand_cost * float(np.mean(nad))
                     + self.acc_margin_bonus * float(np.mean(zapas)))
        self._acc_demand = None     # reset for the next control step

        r_approach = 0.0
        if self.approach_speed_cost:
            v_eef = np.asarray(self.sim.data.get_site_xvelp(self._eef_site_name),
                               dtype=np.float64)
            # TRUE geometry, not the observation: reward must not depend on sensor noise.
            kier = np.asarray(tgt_pos - eef_pos, dtype=np.float64)
            d = dist
            if d > 1e-6:
                # velocity component toward the target (positive = approaching)
                v_do = float(np.dot(v_eef, kier / d))
                v_dop = d / max(self.approach_brake_time, 1e-3)
                nadmiar = v_do - v_dop
                if nadmiar > 0.0:
                    r_approach = -self.approach_speed_cost * nadmiar * nadmiar

        r_action = 0.0
        if action is not None:
            action = np.asarray(action, dtype=np.float64)
            r_action = -self.ACTION_COST * float(np.sum(np.square(action)))
            if self._prev_action is not None:
                delta = action - self._prev_action
                r_action -= self.action_rate_cost * float(np.sum(np.square(delta)))
            self._prev_action = action.copy()

        total = (r_shape + r_dense + r_depress + press_bonus + r_complete
                 + r_collision + r_action + r_finger_misuse + r_contact_speed
                 + r_approach + r_acc)

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

    def _check_spot_collision(self):
        """True if any robot geom touches the Spot body. Counted separately from
        board collisions to keep the metric comparable with runs 6.1-6.41."""
        if not (self.spot_body or self.spot_front_cam):
            return False
        wlasne = ("spot_body_collision", "spot_frontcam_collision")
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            g1 = self.sim.model.geom_id2name(contact.geom1) or ""
            g2 = self.sim.model.geom_id2name(contact.geom2) or ""
            if g1 in wlasne or g2 in wlasne:
                other = g2 if g1 in wlasne else g1
                if other.startswith("robot0") or "gripper0" in other:
                    return True
        return False

    def _check_board_collision(self):
        """True if any robot geom touches the board plate or the wall behind it
        (buttons excluded). The wall counts like the plate (docs 6.19)."""
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            geom1 = self.sim.model.geom_id2name(contact.geom1) or ""
            geom2 = self.sim.model.geom_id2name(contact.geom2) or ""
            robot_geom = any(g.startswith(("robot", "gripper"))
                             for g in (geom1, geom2))
            board_geom = any("board" in g or g == "wall_panel"
                             for g in (geom1, geom2))
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
        """True if an ARM link geom (robot0_*) touches any button. Gripper geoms are
        exempt (v2.3): penalising them made the policy avoid buttons."""
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            g1, g2 = contact.geom1, contact.geom2
            for r, b in ((g1, g2), (g2, g1)):
                if b in self._btn_geom_ids and r in self._arm_geom_ids:
                    return True
        return False

    def _finger_touched_buttons(self):
        """Button indices touched by any gripper finger geom this step (v2.5)."""
        touched = set()
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            g1, g2 = contact.geom1, contact.geom2
            for a, b in ((g1, g2), (g2, g1)):
                if a in self._finger_geom_ids and b in self._geom_to_btn:
                    touched.add(self._geom_to_btn[b])
        return touched

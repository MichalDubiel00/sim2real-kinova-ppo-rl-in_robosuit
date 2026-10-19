#!/usr/bin/env python3
"""
Observation core of the bridge, shared by Webots and hardware (no ROS). Docs 6.56-6.57.

Input: joint angles/velocities, board pose in the arm base frame, target index.
Output: the exact [120] observation the policy saw in training.

FK runs in the training MuJoCo model, not TF: hand-made conventions (grip offset,
site rotation, quaternion sign rule) broke on the arm_d pose. Requires the robot's
joint zeros/directions to match robosuite (checked in Webots, docs 6.26).

Angle unwrapping is critical: j1, j4, j6 are continuous in robosuite (arm_d j6 is
always in [3.14, 6.37]), while ros2_kortex wraps them to [-pi, pi].

Board pose = (p, R) of the cross centre on the back plate, in the arm base frame.
"""

import sys
sys.modules["tensorflow"] = None   # before SB3, otherwise segfault (docs 7.6)

import os

import numpy as np

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _REPO)

# arm_d scene contract (models/README.md, training/configs/arm_d.yaml).
SCENE_ARM_D = dict(
    model=os.path.join(_REPO, "logs/board_6dof/arm_d/checkpoints/ppo_board_98123552_steps.zip"),
    base_x_offset=-0.34,
    init_qpos=[0.0, 1.4692, -2.2410, 3.1403, -2.0200, 3.1416],
    estop_cfg=os.path.join(_REPO, "training/configs/ppo_board_6dof_estop_hw.yaml"),
)

CONTINUOUS = (0, 3, 5)          # j1, j4, j6: unlimited in robosuite and Kortex
# Gripper medians from arm_d episodes (closed throughout training; 6.56).
GRIPPER_QPOS_MED = np.array([0.4786, -0.2359, 0.2425, 0.5619, 0.1558, 0.1560])
GRIPPER_QVEL_MED = np.array([-0.0002, -0.0041, 0.0004, 0.0015, -0.0006, 0.0010])

OBJECT_KEYS = ("target_relative_pos", "target_distance", "target_onehot",
               "target_press_axis", "eef_align_with_target")


def wrap_pi(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


class JointUnwrapper:
    """Continuous branch of the continuous joints, anchored at the start pose.

    reset(q_hw) picks the branch nearest init_qpos; each step then adds the wrapped
    increment (< 0.07 rad at 20 Hz, so a pi jump between samples is impossible).
    """

    def __init__(self, anchor):
        self.anchor = np.asarray(anchor, dtype=np.float64)
        self.q_prev_hw = None
        self.q_prev = None

    def reset(self, q_hw):
        q = np.asarray(q_hw, dtype=np.float64).copy()
        for j in CONTINUOUS:
            q[j] = self.anchor[j] + wrap_pi(q[j] - self.anchor[j])
        self.q_prev_hw = np.asarray(q_hw, dtype=np.float64).copy()
        self.q_prev = q
        return q.copy()

    def __call__(self, q_hw):
        if self.q_prev is None:
            return self.reset(q_hw)
        q_hw = np.asarray(q_hw, dtype=np.float64)
        q = q_hw.copy()
        for j in CONTINUOUS:
            q[j] = self.q_prev[j] + wrap_pi(q_hw[j] - self.q_prev_hw[j])
        self.q_prev_hw, self.q_prev = q_hw.copy(), q
        return q.copy()


class ObsCore:
    """Builds the [120] observation from (q, qd, board pose, target) using the
    training scene's MuJoCo model. The env is used only as a kinematics
    calculator; env.step() is never called."""

    def __init__(self, scene=SCENE_ARM_D):
        import robosuite as suite
        import yaml
        from envs.button_board import ButtonBoardEnv

        c = yaml.safe_load(open(scene["estop_cfg"]))
        ctrl = suite.load_composite_controller_config(controller="BASIC", robot="Kinova3_6DOF")
        self.env = ButtonBoardEnv(
            robots="Kinova3_6DOF", controller_configs=ctrl,
            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
            horizon=2000, random_order=True,
            activation_depth=c.get("activation_depth"),
            force_gripper_closed=True, finger_press_gate=True,
            base_x_offset=scene["base_x_offset"], init_qpos=scene["init_qpos"],
            spot_body=True, spot_front_cam=True,
            button_estop_profile=c["button_estop_profile"],
            board_pos_noise=0.0, base_z_noise=0.0, percep_pos_bias=0.0,
            percep_pos_noise=0.0, percep_axis_noise=0.0,
        )
        self.env.reset()
        env, sim = self.env, self.env.sim
        rob = env.robots[0]
        self.pf = rob.robot_model.naming_prefix
        self.q_idx = np.asarray(rob._ref_joint_pos_indexes)
        self.qd_idx = np.asarray(rob._ref_joint_vel_indexes)
        base_id = sim.model.body_name2id(f"{self.pf}base")
        self.base_in_world = sim.data.body_xpos[base_id].copy()
        assert np.allclose(sim.data.body_xmat[base_id].reshape(3, 3), np.eye(3), atol=1e-6), \
            "arm base rotated in the scene - the core assumes R = I"

        # Socket layout relative to the board cross centre (board frame = world, R = I).
        self.board_center_world = np.array([env.WALL_X, env._board_y, env._board_z])
        tops = np.array([sim.data.site_xpos[i] for i in env._btn_top_ids])
        self.btn_offsets = tops - self.board_center_world
        self.btn_normals = np.array([np.asarray(env.BUTTON_FACES[i][1], dtype=float)
                                     for i in range(env.N_BUTTONS)])
        self.n_buttons = env.N_BUTTONS
        self.init_qpos = np.asarray(scene["init_qpos"], dtype=np.float64)
        self.fk_dt = float(sim.model.opt.timestep)
        self.eef_site = rob.eef_site_id[rob.arms[0]]

        obs = env._get_observations(force_update=True)
        self.keys = list(obs.keys())
        self._check_aggregates(obs)

    # ---------------------------------------------------------------- FK --
    def grip_pose(self, q):
        """(p[3], R[3x3]) of the grip point in the base frame. Wrapped angles are fine."""
        import mujoco
        sim = self.env.sim
        m, d = sim.model._model, sim.data._data
        sim.data.qpos[self.q_idx] = np.asarray(q, dtype=np.float64)
        mujoco.mj_kinematics(m, d)
        return (d.site_xpos[self.eef_site].copy() - self.base_in_world,
                d.site_xmat[self.eef_site].reshape(3, 3).copy())

    # ------------------------------------------------------ jacobian --
    def joint_step(self, q, v, w, dt, damping=0.02):
        """Joint increment realising grip-point twist (v, w) in the base frame over dt:
        DLS on the training-model Jacobian. Local and continuous, unlike MoveIt KDL,
        which returned random flipped branches (15.10)."""
        import mujoco
        sim = self.env.sim
        m, d = sim.model._model, sim.data._data
        sim.data.qpos[self.q_idx] = np.asarray(q, dtype=np.float64)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, self.eef_site)
        J = np.vstack([jp[:, self.qd_idx], jr[:, self.qd_idx]])   # base = world up to rotation
        x = np.r_[np.asarray(v, dtype=np.float64), np.asarray(w, dtype=np.float64)] * dt
        return J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), x)

    # ------------------------------------------- collisions (planning) --
    def collisions(self, q, tol=0.002):
        """Arm/gripper contacts deeper than tol in the training scene (self, Spot
        body, camera, board), excluding adjacent links. Checks homing paths."""
        import mujoco
        sim = self.env.sim
        m, d = sim.model._model, sim.data._data
        if not hasattr(self, "_geom_names"):
            self._geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
        names = self._geom_names
        rob = lambda g: names[g].startswith(("robot0_", "gripper0_"))
        sim.data.qpos[self.q_idx] = np.asarray(q, dtype=np.float64)
        mujoco.mj_forward(m, d)
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if not (rob(g1) or rob(g2)) or c.dist > -tol:
                continue
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            if rob(g1) and rob(g2) and (b1 == b2 or m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1):
                continue
            if names[g1].startswith("gripper0_") and names[g2].startswith("gripper0_"):
                continue
            out.append((names[g1], names[g2], round(float(c.dist), 4)))
        return out

    def safe_home_path(self, q_now, n=60, depart_frac=0.15):
        """Path to init_qpos: straighten the arm (j2 0.3, j3 -1.1, j5 -1.6) ->
        rotate j4, j6 while straight -> init_qpos. A straight joint-space line from
        the Webots start pose self-collides (15.10). Returns (waypoints, report) or
        (None, report). Contacts allowed only at the start of the first segment."""
        q_now = np.asarray(q_now, dtype=np.float64)
        q_now = q_now.copy()
        home = self.init_qpos
        # j4/j6 branch nearest the goal, so the wrist does not spin 2*pi
        for j in CONTINUOUS:
            q_now[j] = home[j] + wrap_pi(q_now[j] - home[j]) if j != 0 else q_now[j]
        W1 = np.array([q_now[0], 0.3, -1.1, q_now[3], -1.6, q_now[5]])
        W2 = np.array([home[0], 0.3, -1.1, home[3], -1.6, home[5]])
        wps = [q_now, W1, W2, home]
        bad = []
        for s_i, (a, b) in enumerate(zip(wps[:-1], wps[1:])):
            for k, t in enumerate(np.linspace(0, 1, n)):
                c = self.collisions((1 - t) * a + t * b)
                if c and not (s_i == 0 and k <= depart_frac * n):
                    bad.append((s_i, k, c[0]))
        if bad:
            return None, f"collision on the path: segment {bad[0][0]}, sample {bad[0][1]}: {bad[0][2]} (+{len(bad) - 1})"
        return [W1, W2, home], "collision-free path (except leaving the start pose)"

    # --------------------------------------------------------- board --
    def nominal_board_pose(self):
        """Noise-free training board pose in the base frame."""
        return self.board_center_world - self.base_in_world, np.eye(3)

    def buttons_base(self, board_p, board_R):
        """(sockets[5,3], normals[5,3]) in the base frame."""
        R = np.asarray(board_R, dtype=np.float64)
        p = np.asarray(board_p, dtype=np.float64)
        return p + self.btn_offsets @ R.T, self.btn_normals @ R.T

    # --------------------------------------------------- observation --
    def build(self, q, qd, board_p, board_R, target_idx, gripper="median", btn_depth=None):
        """q, qd: UNWRAPPED joint angles [6] and velocities [6].
        board_p, board_R: board pose in the base frame. Returns (flat[120], dict).
        btn_depth: [5] cap depression [m]; training targets the live button_top
        site, which sinks up to 6 mm. None = not pressed."""
        sim = self.env.sim
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        # Reproduce a training artifact: eef_pos/eef_quat lag joint_pos by one
        # 2 ms physics substep, so FK uses q - qd*dt and joint blocks use q.
        sim.data.qpos[self.q_idx] = q - qd * self.fk_dt
        sim.data.qvel[self.qd_idx] = qd
        sim.forward()
        obs = self.env._get_observations(force_update=True)
        pf = self.pf
        obs[f"{pf}joint_pos"] = q.copy()
        obs[f"{pf}joint_pos_cos"] = np.cos(q)
        obs[f"{pf}joint_pos_sin"] = np.sin(q)

        # Gripper: training medians (the real 2F-85 has no counterpart of the
        # 6 Robotiq85 joints). A (qpos, qvel) tuple is used as-is, for verification.
        gq, gqv = ((GRIPPER_QPOS_MED, GRIPPER_QVEL_MED) if gripper == "median" else gripper)
        obs[f"{pf}gripper_qpos"] = np.asarray(gq, dtype=np.float64).copy()
        obs[f"{pf}gripper_qvel"] = np.asarray(gqv, dtype=np.float64).copy()

        # Object part: same formulas as ButtonBoardEnv._setup_observables, with
        # the socket from perception instead of the simulator.
        btn, nrm = self.buttons_base(board_p, board_R)
        if btn_depth is not None:
            btn = btn - nrm * np.asarray(btn_depth, dtype=np.float64)[:, None]
        eef_base = np.asarray(obs[f"{pf}eef_pos"]) - self.base_in_world
        rel = btn[target_idx] - eef_base
        axis = -nrm[target_idx]
        axis = axis / np.linalg.norm(axis)
        oh = np.zeros(self.n_buttons, dtype=np.float32)
        oh[target_idx] = 1.0
        eef_z = _quat2mat(np.asarray(obs[f"{pf}eef_quat"]))[:, 2]
        align = np.arccos(np.clip(abs(float(np.dot(eef_z, axis))), -1.0, 1.0))
        obs["target_relative_pos"] = rel
        obs["target_distance"] = np.array([np.linalg.norm(rel)])
        obs["target_onehot"] = oh
        obs["target_press_axis"] = axis.astype(np.float32)
        obs["eef_align_with_target"] = np.array([align])

        self._rebuild_aggregates(obs)
        flat = np.concatenate([np.asarray(obs[k]).ravel() for k in self.keys]).astype(np.float32)
        return flat, obs

    # robosuite builds "*-state" aggregates in registration order.
    def _modality_keys(self, agg):
        mod = agg[:-len("-state")]
        obsv = self.env._observables
        return [k for k in self.keys if k in obsv and obsv[k].modality == mod]

    def _check_aggregates(self, obs):
        for agg in [k for k in self.keys if k.endswith("-state")]:
            parts = self._modality_keys(agg)
            cat = np.concatenate([np.asarray(obs[k]).ravel() for k in parts])
            assert np.allclose(cat, np.asarray(obs[agg]).ravel()), \
                f"aggregate {agg} is not a concatenation of {parts}"
        self.agg_parts = {agg: self._modality_keys(agg)
                          for agg in self.keys if agg.endswith("-state")}

    def _rebuild_aggregates(self, obs):
        for agg, parts in self.agg_parts.items():
            obs[agg] = np.concatenate([np.asarray(obs[k]).ravel() for k in parts])


def _quat2mat(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])

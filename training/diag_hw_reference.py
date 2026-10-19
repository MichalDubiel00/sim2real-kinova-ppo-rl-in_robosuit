#!/usr/bin/env python3
"""
robosuite reference values for the hardware bridge (arm_d scene).
  1. nominal geometry: arm base in world, sockets/normals relative to the base, TCP at start
  2. action -> TCP VELOCITY gain under the training limits
  3. gripper medians (closed) from policy episodes

  python training/diag_hw_reference.py --model logs/board_6dof/arm_d/checkpoints/ppo_board_98123552_steps.zip
"""

import sys
sys.modules["tensorflow"] = None   # before SB3, otherwise segfault (docs 7.6)

import argparse
import json
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import robosuite as suite
import robosuite.utils.transform_utils as T
import yaml
from envs.button_board import ButtonBoardEnv

INIT_QPOS = [0.0, 1.4692, -2.2410, 3.1403, -2.0200, 3.1416]
KINOVA_JOINT_VEL = [1.39, 1.39, 1.39, 1.22, 1.22, 1.22]
KINOVA_TWIST = (0.5, 0.8727)
HW_JOINT_ACC = [5.2, 5.2, 5.2, 10.0, 10.0, 10.0]
ESTOP_CFG = "training/configs/ppo_board_6dof_estop_hw.yaml"


def make_env(nominal, horizon=2000):
    c = yaml.safe_load(open(ESTOP_CFG))
    noise = dict(board_pos_noise=0.0, base_z_noise=0.0, percep_pos_bias=0.0,
                 percep_pos_noise=0.0, percep_axis_noise=0.0) if nominal else {}
    ctrl = suite.load_composite_controller_config(controller="BASIC", robot="Kinova3_6DOF")
    return ButtonBoardEnv(
        robots="Kinova3_6DOF", controller_configs=ctrl,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        horizon=horizon, random_order=True,
        activation_depth=c.get("activation_depth"),
        force_gripper_closed=True, finger_press_gate=True,
        joint_acc_limits=HW_JOINT_ACC, joint_acc_limit_mode="scale",
        joint_vel_limits=KINOVA_JOINT_VEL, joint_vel_limit_mode="scale",
        eef_twist_limits=KINOVA_TWIST,
        base_x_offset=-0.34, init_qpos=INIT_QPOS,
        spot_body=True, spot_front_cam=True,
        button_estop_profile=c["button_estop_profile"],
        **noise,
    )


def geometry(env):
    env.reset()
    sim = env.sim
    pf = env.robots[0].robot_model.naming_prefix
    base_id = sim.model.body_name2id(f"{pf}base")
    base_pos = sim.data.body_xpos[base_id].copy()
    base_R = sim.data.body_xmat[base_id].reshape(3, 3).copy()
    eef_id = env.robots[0].eef_site_id[env.robots[0].arms[0]]
    eef = sim.data.site_xpos[eef_id].copy()
    eef_R = sim.data.site_xmat[eef_id].reshape(3, 3)
    obs = env._get_observations(force_update=True)
    btn = [sim.data.site_xpos[i].copy() for i in env._btn_top_ids]
    nrm = [np.asarray(env.BUTTON_FACES[i][1], dtype=float) for i in range(env.N_BUTTONS)]
    out = {
        "base_in_world": base_pos.tolist(),
        "base_rot_is_identity": bool(np.allclose(base_R, np.eye(3), atol=1e-6)),
        "qpos": env.robots[0]._joint_positions.tolist(),
        "eef_pos_world": eef.tolist(),
        "eef_pos_base": (base_R.T @ (eef - base_pos)).tolist(),
        "eef_quat_obs": obs[f"{pf}eef_quat"].tolist(),
        "eef_z_axis": eef_R[:, 2].tolist(),
        "buttons_base": [(base_R.T @ (b - base_pos)).tolist() for b in btn],
        "normals_world": [n.tolist() for n in nrm],
    }
    return out


def gains(env, n_steps=30, tail=10):
    """Constant +-amp action on one axis; TCP velocity over the last `tail` steps."""
    dt = 1.0 / env.control_freq
    eef_id = env.robots[0].eef_site_id[env.robots[0].arms[0]]
    rows = []
    for axis in range(6):
        for amp in (0.25, 0.5, 1.0):
            for sgn in (+1, -1):
                env.reset()
                a = np.zeros(env.action_dim)
                a[axis] = sgn * amp
                P, R = [], []
                for _ in range(n_steps):
                    env.step(a)
                    P.append(env.sim.data.site_xpos[eef_id].copy())
                    R.append(env.sim.data.site_xmat[eef_id].reshape(3, 3).copy())
                P, R = np.array(P), np.array(R)
                if axis < 3:
                    v = (P[-1] - P[-tail - 1]) / (tail * dt)
                    along = float(v[axis]) * sgn
                    cross = float(np.linalg.norm(np.delete(v, axis)))
                else:
                    # angular velocity in world: log(R_end R_start^T) / time
                    dR = R[-1] @ R[-tail - 1].T
                    w = T.quat2axisangle(T.mat2quat(dR)) / (tail * dt)
                    along = float(w[axis - 3]) * sgn
                    cross = float(np.linalg.norm(np.delete(w, axis - 3)))
                rows.append(dict(axis=axis, amp=amp, sgn=sgn,
                                 v_along=along, v_cross=cross,
                                 gain=along / amp))
    return rows


def policy_rollouts(model_path, episodes=6, seed=1234, raw_path=None):
    """Policy episodes under training conditions: gripper medians and (action,
    realised TCP velocity next step) pairs at the operating point."""
    from stable_baselines3 import PPO
    env = make_env(nominal=False)
    model = PPO.load(model_path, device="cpu")
    pf = env.robots[0].robot_model.naming_prefix
    eef_id = env.robots[0].eef_site_id[env.robots[0].arms[0]]
    dt = 1.0 / env.control_freq
    q, qv, A, V, W = [], [], [], [], []
    for ep in range(episodes):
        np.random.seed(seed + ep)
        obs = env.reset()
        for _ in range(2000):
            p0 = env.sim.data.site_xpos[eef_id].copy()
            R0 = env.sim.data.site_xmat[eef_id].reshape(3, 3).copy()
            a, _ = model.predict(flatten_obs(obs), deterministic=True)
            obs, _, done, info = env.step(a)
            p1 = env.sim.data.site_xpos[eef_id].copy()
            R1 = env.sim.data.site_xmat[eef_id].reshape(3, 3).copy()
            A.append(np.asarray(a[:6], dtype=float))
            V.append((p1 - p0) / dt)
            W.append(T.quat2axisangle(T.mat2quat(R1 @ R0.T)) / dt)
            q.append(obs[f"{pf}gripper_qpos"]); qv.append(obs[f"{pf}gripper_qvel"])
            if done or info.get("success"):
                break
    env.close()
    A, V, W = np.array(A), np.array(V), np.array(W)
    if raw_path:
        np.savez(raw_path, A=A, V=V, W=W, dt=dt)
    fit = []
    for i in range(6):
        y = V[:, i] if i < 3 else W[:, i - 3]
        x = A[:, i]
        k = float(np.dot(x, y) / np.dot(x, x))            # y = k*x, no intercept
        r2 = 1.0 - float(np.sum((y - k * x) ** 2) / np.sum((y - y.mean()) ** 2))
        fit.append(dict(axis=i, k=k, r2=r2, n=len(x),
                        sat=float(np.mean(np.abs(x) > 0.99))))
    return (np.median(q, axis=0).tolist(), np.median(qv, axis=0).tolist(), fit)


def flatten_obs(obs_dict):
    """Same as compare_models.flatten_obs."""
    return np.concatenate([np.asarray(v).flatten() for v in obs_dict.values()],
                          dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--json", default=None, help="write results to a file")
    args = ap.parse_args()

    env = make_env(nominal=True)
    print(f"control_freq {env.control_freq} Hz")
    g = geometry(env)
    np.set_printoptions(precision=4, suppress=True)
    print("\n=== NOMINAL GEOMETRY (no noise) ===")
    for k, v in g.items():
        print(f"  {k:22s} {np.round(np.asarray(v), 4).tolist() if not isinstance(v, bool) else v}")

    print("\n=== GAIN action -> TCP velocity (last 10 of 30 steps) ===")
    print(f"  {'os':>3} {'amp':>5} {'znak':>4} {'v wzdluz':>9} {'v w bok':>8} {'v/amp':>7}")
    rows = gains(env)
    names = ["X", "Y", "Z", "RX", "RY", "RZ"]
    for r in rows:
        print(f"  {names[r['axis']]:>3} {r['amp']:5.2f} {r['sgn']:+4d} "
              f"{r['v_along']:9.4f} {r['v_cross']:8.4f} {r['gain']:7.4f}")
    env.close()

    res = {"geometry": g, "gains": rows}
    if args.model:
        gq, gqv, fit = policy_rollouts(
            args.model, raw_path=(args.json.replace('.json', '_raw.npz') if args.json else None))
        print(f"\n=== GRIPPER (medians from policy episodes) ===\n  qpos {np.round(gq,4)}\n  qvel {np.round(gqv,4)}")
        print("\n=== OPERATING POINT: TCP velocity ~ k * action (regression on policy steps) ===")
        print(f"  {'ax':>3} {'k':>8} {'R2':>6} {'sat.':>7}   (k in m/s or rad/s per action unit)")
        for f in fit:
            print(f"  {names[f['axis']]:>3} {f['k']:8.4f} {f['r2']:6.3f} {100*f['sat']:6.0f}%   n={f['n']}")
        res["gripper_qpos_med"], res["gripper_qvel_med"], res["operating_fit"] = gq, gqv, fit
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)
        print(f"\nsaved {args.json}")


if __name__ == "__main__":
    main()

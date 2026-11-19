#!/usr/bin/env python3
"""
Action -> grip-point twist filter for the hardware bridge, fitted in the training scene.

Runs the policy in the training config (noise off, no delay), records the in-loop EMA output
e[t] and the mean grip-point twist over each control step, and fits per axis
    v[t] = alpha * v[t-1] + beta * e[t]
(capped at eef_twist_limits). Also reports gripper medians (observation units) and how often
the twist falls into the lab driver's dead band (all |components| < 0.025 m/s and rad/s).

  python training/fit_twist_filter.py --config training/configs/arm_ps.yaml \
      --model logs/board_6dof/arm_p8f/checkpoints/ppo_board_130123552_steps.zip --episodes 10
"""

import sys
sys.modules["tensorflow"] = None   # before SB3, otherwise segfault (docs 7.6)

import argparse
import json
import os

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_board import env_kwargs_from_cfg
from envs.button_board import ButtonBoardEnv

NOMINAL = dict(board_pos_noise=0.0, percep_pos_bias=0.0, percep_pos_noise=0.0, percep_axis_noise=0.0,
               base_z_noise=0.0, base_z_range=None, base_x_noise=0.0, action_delay_steps=None,
               gain_noise_lin=None, gain_noise_rot=None, joint_acc_limits_noise=None)
DEADBAND = 0.025   # lab driver: message < 0.05 on every component, message = 2 x command


def rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def fit_axis(v, e, first):
    """Least squares v[t] = a v[t-1] + b e[t], skipping episode starts."""
    keep = ~first
    X = np.c_[np.r_[0.0, v[:-1]], e][keep]
    y = v[keep]
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ np.array([a, b])
    r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    return float(a), float(b), float(r2)


def simulate(alpha, beta, e, first, v_cap, w_cap):
    x = np.zeros(6)
    out = np.zeros_like(e)
    for t in range(len(e)):
        if first[t]:
            x = np.zeros(6)
        x = alpha * x + beta * e[t]
        for sl, cap in ((slice(0, 3), v_cap), (slice(3, 6), w_cap)):
            n = np.linalg.norm(x[sl])
            if n > cap:
                x[sl] *= cap / n
        out[t] = x
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="JSON with alpha, beta, caps, gripper medians")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    cfg = yaml.safe_load(open(args.config))
    kw = env_kwargs_from_cfg(cfg)
    kw.update(NOMINAL)
    env = ButtonBoardEnv(**kw)
    model = PPO.load(args.model, device="cpu")
    ema = float(cfg.get("action_ema") or 1.0)
    T = 1.0 / float(cfg.get("control_freq", 20))
    v_cap, w_cap = cfg.get("eef_twist_limits") or (0.5, 0.8727)
    sid = env.robots[0].eef_site_id["right"]
    rng = np.random.default_rng(args.seed)

    E, V, F, GQ, GV, NEAR, OK = [], [], [], [], [], [], []
    for ep in range(args.episodes):
        np.random.seed(int(rng.integers(1 << 30)))
        obs = env.reset()
        keys = list(obs.keys())
        flat = lambda o: np.concatenate([np.asarray(o[k]).ravel() for k in keys]).astype(np.float32)
        e = np.zeros(6)
        p0 = env.sim.data.site_xpos[sid].copy(); R0 = env.sim.data.site_xmat[sid].reshape(3, 3).copy()
        first = True
        for t in range(int(cfg.get("horizon", 2000))):
            a, _ = model.predict(flat(obs), deterministic=True)
            e = ema * np.clip(a[:6], -1, 1) + (1 - ema) * e
            obs, r, done, info = env.step(a)
            p1 = env.sim.data.site_xpos[sid].copy(); R1 = env.sim.data.site_xmat[sid].reshape(3, 3).copy()
            v = (p1 - p0) / T
            w = rotvec(R1 @ R0.T) / T
            E.append(e.copy()); V.append(np.r_[v, w]); F.append(first)
            GQ.append(np.asarray(obs["robot0_gripper_qpos"])); GV.append(np.asarray(obs["robot0_gripper_qvel"]))
            NEAR.append(float(np.ravel(obs["target_distance"])[0]))
            p0, R0, first = p1, R1, False
            if done:
                break
        OK.append(t + 1)
        print(f"episode {ep}: {t + 1} steps")
    E, V, F, NEAR = np.array(E), np.array(V), np.array(F), np.array(NEAR)

    names = ["vx", "vy", "vz", "wx", "wy", "wz"]
    alpha, beta = np.zeros(6), np.zeros(6)
    print(f"\n{len(E)} steps at {1 / T:.3f} Hz, EMA {ema}, caps {v_cap} m/s {w_cap} rad/s")
    print(f"{'axis':>4} {'alpha':>7} {'beta':>8} {'gain':>7} {'R2':>6}")
    for i in range(6):
        a_, b_, r2 = fit_axis(V[:, i], E[:, i], F)
        alpha[i], beta[i] = a_, b_
        print(f"{names[i]:>4} {a_:7.3f} {b_:8.4f} {b_ / max(1 - a_, 1e-6):7.3f} {r2:6.3f}")
    P = simulate(alpha, beta, E, F, v_cap, w_cap)
    for i in range(6):
        r2 = 1 - np.sum((V[:, i] - P[:, i]) ** 2) / np.sum((V[:, i] - V[:, i].mean()) ** 2)
        print(f"  open-loop simulated filter, {names[i]}: R2 {r2:.3f}")

    lin_small = np.all(np.abs(P[:, :3]) < DEADBAND, axis=1)
    ang_small = np.all(np.abs(P[:, 3:]) < DEADBAND, axis=1)
    db = lin_small & ang_small
    near = NEAR < 0.05
    print(f"\ndead band (all |v|,|w| components < {DEADBAND}): {100 * db.mean():.1f} % of steps; "
          f"within 5 cm of the target {100 * db[near].mean():.1f} % ({near.sum()} steps)")
    sp = np.linalg.norm(V[near, :3], axis=1)
    print(f"realised grip speed within 5 cm of the target: median {sp.mean():.3f}, "
          f"p10 {np.percentile(sp, 10):.3f}, p90 {np.percentile(sp, 90):.3f} m/s")

    gq, gv = np.median(np.array(GQ), axis=0), np.median(np.array(GV), axis=0)
    print(f"gripper qpos median {np.round(gq, 4).tolist()}\ngripper qvel median {np.round(gv, 4).tolist()}")
    if args.out:
        np.savez(os.path.splitext(args.out)[0] + "_raw.npz", e=E, v=V, first=F, near=NEAR)
        json.dump(dict(config=args.config, model=args.model, control_freq=1 / T, action_ema=ema,
                       alpha=alpha.tolist(), beta=beta.tolist(), v_cap=v_cap, w_cap=w_cap,
                       gripper_qpos_med=gq.tolist(), gripper_qvel_med=gv.tolist(),
                       deadband_frac=float(db.mean()), deadband_frac_near=float(db[near].mean())),
                  open(args.out, "w"), indent=1)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()

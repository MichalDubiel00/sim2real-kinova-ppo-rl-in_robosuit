"""
What collision steps really mean for sim2real: contact FORCE [N] (protective-stop
risk), WHICH geom (arm link vs gripper housing), contact DURATION, and whether it is
a side effect of pressing or driving into the plate/wall.

    python training/diag_collision_forces.py --model <zip> --episodes 20 --horizon 2000
"""

import argparse
import os
import sys
from collections import defaultdict

sys.modules["tensorflow"] = None  # docs 7.6

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import robosuite as suite
from envs.button_board import ButtonBoardEnv


def contact_forces(env):
    """[(force_N, robot_geom_name, is_gripper), ...] for robot <-> plate/wall contacts
    this step (buttons skipped)."""
    m = env.sim.model._model if hasattr(env.sim.model, "_model") else env.sim.model
    d = env.sim.data._data if hasattr(env.sim.data, "_data") else env.sim.data
    out = []
    buf = np.zeros(6, dtype=np.float64)
    for i in range(d.ncon):
        c = d.contact[i]
        n1 = env.sim.model.geom_id2name(c.geom1) or ""
        n2 = env.sim.model.geom_id2name(c.geom2) or ""
        pair = ((n1, n2), (n2, n1))
        for a, b in pair:
            robot = a.startswith(("robot", "gripper"))
            board = ("board" in b) or (b == "wall_panel")
            if robot and board:
                mujoco.mj_contactForce(m, d, i, buf)
                out.append((float(np.linalg.norm(buf[:3])), a,
                            a.startswith("gripper")))
                break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="logs/board_6dof/s2/best_model/best_model.zip")
    p.add_argument("--robot", default="Kinova3_6DOF")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--horizon", type=int, default=2000)
    p.add_argument("--free-gripper", action="store_true")
    args = p.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    ctrl = suite.load_composite_controller_config(controller="BASIC",
                                                  robot=args.robot)
    env = ButtonBoardEnv(
        robots=args.robot, controller_configs=ctrl,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        horizon=args.horizon, random_order=True, board_pos_noise=0.05,
        force_gripper_closed=not args.free_gripper,
    )

    forces_arm, forces_grip = [], []
    per_geom = defaultdict(int)
    streaks, cur = [], 0
    succ = 0

    for ep in range(args.episodes):
        np.random.seed(1234 + ep)
        obs = env.reset()
        done = False
        while not done:
            flat = np.concatenate([np.asarray(v).flatten()
                                   for v in obs.values()], dtype=np.float32)
            a, _ = model.predict(flat, deterministic=True)
            obs, _, done, _ = env.step(a)
            cs = contact_forces(env)
            if cs:
                cur += 1
                for f, g, is_grip in cs:
                    (forces_grip if is_grip else forces_arm).append(f)
                    per_geom[g] += 1
            else:
                if cur:
                    streaks.append(cur)
                cur = 0
        if cur:
            streaks.append(cur)
        cur = 0
        succ += int(all(env._pressed))

    def stat(v, label):
        if not v:
            print(f"  {label:22s} no contacts")
            return
        v = np.array(v)
        print(f"  {label:22s} n={len(v):5d}  median {np.median(v):6.2f} N  "
              f"p90 {np.percentile(v,90):7.2f} N  max {v.max():8.2f} N")

    print(f"\nmodel: {args.model}")
    print(f"episodes {args.episodes}, horizon {args.horizon}, "
          f"successes {succ}/{args.episodes}\n")
    print("CONTACT FORCES with plate/wall:")
    stat(forces_arm, "ARM LINKS")
    stat(forces_grip, "gripper housing")
    allf = np.array(forces_arm + forces_grip)
    if allf.size:
        for prog in (10, 25, 50, 100):
            print(f"  contacts > {prog:3d} N: {int((allf>prog).sum()):5d} "
                  f"({100*(allf>prog).mean():.1f} %)")
    if streaks:
        s = np.array(streaks)
        print(f"\nContact CONTINUITY (consecutive steps, 20 Hz):")
        print(f"  runs {len(s)}, median {int(np.median(s))} st., "
              f"p90 {int(np.percentile(s,90))} kr., max {int(s.max())} kr. "
              f"({s.max()/20:.1f} s)")
    print("\nMOST FREQUENT CONTACT GEOMS:")
    for g, n in sorted(per_geom.items(), key=lambda x: -x[1])[:6]:
        print(f"  {n:6d}  {g}")


if __name__ == "__main__":
    main()

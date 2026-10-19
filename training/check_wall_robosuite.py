#!/usr/bin/env python3
"""
Does the virtual wall remove motion the policy needs to press? (docs 6.62)
arm_d in the training scene with a real board, paired seeds, without / with
safety.VirtualWall applied to the translation part of the OSC action.

Criterion (fixed before the run): at most 2/40 successes lost, no STOP.
Result 19.10, three attempts all failed (33/40 with 7 STOP; 36/40 with 3 STOP twice):
use the wall only for the in-air test.

  python training/check_wall_robosuite.py --episodes 40
"""

import sys
sys.modules["tensorflow"] = None

import argparse
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ros2_bridge"))

from diag_hw_reference import make_env, flatten_obs
from obs_core import ObsCore, SCENE_ARM_D
from safety import wall_for


def run(model, core, seed, use_wall, margin):
    env = make_env(nominal=False)
    rob = env.robots[0]
    eef = rob.eef_site_id[rob.arms[0]]
    np.random.seed(seed)
    obs = env.reset()
    board_p = np.array([env.WALL_X, env._board_y, env._board_z]) - core.base_in_world
    wall = wall_for(core, board_p, np.eye(3), x_wall=margin) if use_wall else None
    stop, clipped, steps = False, 0, 0
    for steps in range(1, 2001):
        a, _ = model.predict(flatten_obs(obs), deterministic=True)
        if wall is not None:
            grip = env.sim.data.site_xpos[eef] - core.base_in_world
            # translation action ~ motion direction
            v, stop, d = wall.filter(grip, a[:3] * 0.3)       # 0.3 m/s ~ typical speed
            clipped += int(d == "clipped")
            a = a.copy()
            a[:3] = v / 0.3
            if stop:
                break
        obs, _, done, _ = env.step(a)
        if done:
            break
    n = int(sum(env._pressed))
    env.close()
    return n, steps, clipped, stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--margin", type=float, default=-0.015, help="wall x in the board frame [m]")
    args = ap.parse_args()
    from stable_baselines3 import PPO
    model = PPO.load(SCENE_ARM_D["model"], device="cpu")
    core = ObsCore()
    res = {False: [], True: []}
    for ep in range(args.episodes):
        for w in (False, True):
            res[w].append(run(model, core, args.seed + ep, w, args.margin))
        a, b = res[False][-1], res[True][-1]
        print(f"  ep{ep:02d}: no wall {a[0]}/5 {a[1]} st. | wall {b[0]}/5 {b[1]} st., "
              f"clipped {b[2]} st.{' STOP' if b[3] else ''}", flush=True)
    for w in (False, True):
        r = res[w]
        print(f"{'with wall ' if w else 'no wall   '}: successes {sum(1 for x in r if x[0] == 5)}/{len(r)}, "
              f"mean presses {np.mean([x[0] for x in r]):.2f}, median steps {int(np.median([x[1] for x in r]))}"
              + (f", mean clipped steps {np.mean([x[2] for x in r]):.1f}, STOP {sum(x[3] for x in r)}" if w else ""))
    s0 = np.array([x[0] == 5 for x in res[False]]); s1 = np.array([x[0] == 5 for x in res[True]])
    print(f"pairs: only without wall {int(np.sum(s0 & ~s1))}, only with wall {int(np.sum(~s0 & s1))}")


if __name__ == "__main__":
    main()

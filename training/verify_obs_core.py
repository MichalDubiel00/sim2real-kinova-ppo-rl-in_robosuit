#!/usr/bin/env python3
"""
Offline verification of the bridge core (B0/B1 for the arm_d scene, docs 6.57).
arm_d plays ButtonBoardEnv without perception/base-height noise. Each step the core
gets what ROS will give: wrapped joint angles, joint velocities, board pose, target.
B0: |obs_core - obs_env| per block. B1: |action_core - action_env|.
Gripper checked with env values (core test) and with medians (what runs on hardware).

  python training/verify_obs_core.py --episodes 3
"""

import sys
sys.modules["tensorflow"] = None

import argparse
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ros2_bridge"))

from obs_core import ObsCore, JointUnwrapper, SCENE_ARM_D, wrap_pi
from diag_hw_reference import make_env, flatten_obs

BLOCKS = [("joint_pos", 0, 6), ("joint_pos_cos", 6, 12), ("joint_pos_sin", 12, 18),
          ("joint_vel", 18, 24), ("eef_pos", 24, 27), ("eef_quat", 27, 31),
          ("eef_quat_site", 31, 35), ("gripper_qpos", 35, 41), ("gripper_qvel", 41, 47),
          ("target_rel", 47, 50), ("target_dist", 50, 51), ("onehot", 51, 56),
          ("press_axis", 56, 59), ("align", 59, 60)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(SCENE_ARM_D["model"], device="cpu")
    core = ObsCore()

    env = make_env(nominal=False)
    env.base_z_noise = 0.0
    env.percep_pos_bias = env.percep_pos_noise = env.percep_axis_noise = 0.0
    pf = env.robots[0].robot_model.naming_prefix

    print(f"obs keys: {core.keys}")
    err_blk = {n: 0.0 for n, _, _ in BLOCKS}
    err_med_act, err_act, n_steps, n_wrapped, max_unwrap = 0.0, 0.0, 0, 0, 0.0
    err_dep, max_depth = 0.0, 0.0
    for ep in range(args.episodes):
        np.random.seed(args.seed + ep)
        obs = env.reset()
        assert flatten_obs(obs).shape == (120,)
        unw = JointUnwrapper(SCENE_ARM_D["init_qpos"])
        first = True
        for _ in range(2000):
            q_true = np.asarray(obs[f"{pf}joint_pos"], dtype=np.float64)
            q_hw = wrap_pi(q_true)                       # as ros2_kortex reports them
            n_wrapped += int(np.any(np.abs(q_hw - q_true) > 1e-9))
            q = unw.reset(q_hw) if first else unw(q_hw)
            first = False
            max_unwrap = max(max_unwrap, float(np.max(np.abs(q - q_true))))

            board_p = np.array([env.WALL_X, env._board_y, env._board_z]) - core.base_in_world
            # target as observed: the env advances the target AFTER computing obs
            tgt = int(np.argmax(obs["target_onehot"]))
            f_env = flatten_obs(obs)
            f_core, _ = core.build(q, obs[f"{pf}joint_vel"], board_p, np.eye(3), tgt,
                                   gripper=(obs[f"{pf}gripper_qpos"], obs[f"{pf}gripper_qvel"]))
            f_med, _ = core.build(q, obs[f"{pf}joint_vel"], board_p, np.eye(3), tgt)
            # true cap depression: checks whether the rest of the difference is travel
            tops = np.array([env.sim.data.site_xpos[i] for i in env._btn_top_ids]) - core.base_in_world
            nom, nrm = core.buttons_base(board_p, np.eye(3))
            depth = -np.sum((tops - nom) * nrm, axis=1)
            f_dep, _ = core.build(q, obs[f"{pf}joint_vel"], board_p, np.eye(3), tgt,
                                  gripper=(obs[f"{pf}gripper_qpos"], obs[f"{pf}gripper_qvel"]),
                                  btn_depth=depth)
            err_dep = max(err_dep, float(np.max(np.abs(f_dep - f_env))))
            max_depth = max(max_depth, float(depth.max()))
            for n, a, b in BLOCKS:
                d = max(np.max(np.abs(f_core[a:b] - f_env[a:b])),
                        np.max(np.abs(f_core[60 + a:60 + b] - f_env[60 + a:60 + b])))
                err_blk[n] = max(err_blk[n], float(d))
            a_env, _ = model.predict(f_env, deterministic=True)
            a_core, _ = model.predict(f_core, deterministic=True)
            a_med, _ = model.predict(f_med, deterministic=True)
            err_act = max(err_act, float(np.max(np.abs(a_core - a_env))))
            err_med_act += float(np.mean(np.abs(a_med - a_env)))
            n_steps += 1

            obs, _, done, info = env.step(a_env)
            if done or info.get("success"):
                break
        print(f"  ep{ep}: {n_steps} steps total, pressed={int(np.sum(env._pressed)) if hasattr(env, '_pressed') else '?'}, info={ {k: v for k, v in info.items() if np.isscalar(v)} }")

    print("\n=== B0: max |obs_core - obs_env| per block (gripper from env) ===")
    for n, _, _ in BLOCKS:
        flag = "" if err_blk[n] < 1e-4 else "   <-- DIFFERENCE"
        print(f"  {n:14s} {err_blk[n]:.2e}{flag}")
    print(f"\nsteps with a wrapped angle: {n_wrapped}/{n_steps}; "
          f"max unwrap error {max_unwrap:.2e} rad")
    print(f"with true cap depression (max {1000*max_depth:.1f} mm): "
          f"max |obs_core - obs_env| = {err_dep:.2e}")
    print(f"=== B1: max |a_core - a_env| = {err_act:.2e} (nominal caps) ===")
    print(f"gripper from MEDIANS (hardware): mean |a_med - a_env| = {err_med_act / n_steps:.4f}")


if __name__ == "__main__" and "--closed-loop" not in sys.argv:
    main()


def closed_loop(n, seed0=2000, sources=("env", "rdzen")):
    """Closed loop: the policy acts on the CORE observation (wrapped and unwrapped
    angles, nominal caps, median gripper) vs the env observation, same seeds."""
    from stable_baselines3 import PPO
    model = PPO.load(SCENE_ARM_D["model"], device="cpu")
    core = ObsCore()
    env = make_env(nominal=False)
    env.base_z_noise = 0.0
    env.percep_pos_bias = env.percep_pos_noise = env.percep_axis_noise = 0.0
    pf = env.robots[0].robot_model.naming_prefix
    res = {src: [] for src in sources}
    for src in sources:
        for ep in range(n):
            np.random.seed(seed0 + ep)
            obs = env.reset()
            unw = JointUnwrapper(SCENE_ARM_D["init_qpos"])
            first, steps = True, 0
            for steps in range(1, 2001):
                if src == "env":
                    f = flatten_obs(obs)
                else:
                    q_hw = wrap_pi(obs[f"{pf}joint_pos"])
                    q = unw.reset(q_hw) if first else unw(q_hw)
                    first = False
                    board_p = np.array([env.WALL_X, env._board_y, env._board_z]) - core.base_in_world
                    # "rdzen" = median gripper (hardware); "rdzen_chw" = env gripper
                    g = ((obs[f"{pf}gripper_qpos"], obs[f"{pf}gripper_qvel"])
                         if src == "rdzen_chw" else "median")
                    f, _ = core.build(q, obs[f"{pf}joint_vel"], board_p, np.eye(3),
                                      int(np.argmax(obs["target_onehot"])), gripper=g)
                a, _ = model.predict(f, deterministic=True)
                obs, _, done, _ = env.step(a)
                if done:
                    break
            pressed = int(np.sum(env._pressed))
            res[src].append((pressed, steps))
            print(f"  {src:5s} ep{ep:02d}: {pressed}/5, {steps} steps", flush=True)
    for src, r in res.items():
        r = np.array(r)
        print(f"{src:5s}: successes {int(np.sum(r[:, 0] == 5))}/{n}, mean presses {r[:, 0].mean():.2f}, "
              f"median steps {int(np.median(r[:, 1]))}")


if __name__ == "__main__" and "--closed-loop" in sys.argv:
    _src = (sys.argv[sys.argv.index("--sources") + 1].split(",")
            if "--sources" in sys.argv else ("env", "rdzen"))
    _seed = int(sys.argv[sys.argv.index("--seed0") + 1]) if "--seed0" in sys.argv else 2000
    closed_loop(int(sys.argv[sys.argv.index("--closed-loop") + 1]), _seed, _src)

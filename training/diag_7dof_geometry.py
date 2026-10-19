"""
Why does 7DOF v2.3 (90/100 historically) score 0/100 on the current geometry?
Hypothesis: proprioception (absolute joint angles / EEF position) is out of
distribution after the base and board moved. Runs the same model on four geometries:
  A. current                      B. old (control)
  C. roll reverted only           D. heights reverted only
    python training/diag_7dof_geometry.py [--episodes 25]
"""

import argparse
import os
import sys

sys.modules["tensorflow"] = None  # docs 7.6

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import robosuite as suite
from envs.button_board import ButtonBoardEnv, _rolled_button_faces


def make_variant(name, board_z, base_z, roll_deg):
    """Subclass with swapped geometry; BUTTON_FACES is recomputed (computed once at class level)."""
    return type(
        f"ButtonBoardEnv_{name}",
        (ButtonBoardEnv,),
        {
            "BOARD_Z": board_z,
            "KINOVA_BASE_Z": base_z,
            "BOARD_ROLL_DEG": roll_deg,
            "BUTTON_FACES": _rolled_button_faces(roll_deg),
        },
    )


# KINOVA_BASE_Z is now an ABSOLUTE height (bottom_offset subtracted); the old code
# effectively placed the base at 1.562 m, so pass 1.562 to reproduce it.
BASE_STARA = 0.65 + 0.912
BASE_NOWA = 0.65

WARIANTY = [
    ("A_current",        0.60, BASE_NOWA,  45.0),
    ("B_old",         1.20, BASE_STARA,  0.0),
    ("C_roll_only",    0.60, BASE_NOWA,   0.0),
    ("D_heights_only",   1.20, BASE_STARA, 45.0),
]


def run(env, model, episodes, base_seed=1234):
    succ = 0
    pressed_tot = 0
    for ep in range(episodes):
        np.random.seed(base_seed + ep)
        obs = env.reset()
        done = False
        while not done:
            flat = np.concatenate(
                [np.asarray(v).flatten() for v in obs.values()],
                dtype=np.float32)
            action, _ = model.predict(flat, deterministic=True)
            obs, _, done, _ = env.step(action)
        succ += int(all(env._pressed))
        pressed_tot += int(sum(env._pressed))
    return succ, pressed_tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="logs/board/ppo_board_v23_12M.zip")
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--robot", default="Kinova3")
    p.add_argument("--horizon", type=int, default=1000)
    args = p.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")

    ctrl = suite.load_composite_controller_config(
        controller="BASIC", robot=args.robot)

    print(f"model: {args.model}")
    print(f"episodes per variant: {args.episodes}, horizon {args.horizon}\n")
    print(f"{'wariant':16s} {'board_z':>8s} {'baza':>7s} {'roll':>5s} "
          f"{'sukces':>8s} {'wcisniec/ep':>12s}")

    for name, bz, basez, roll in WARIANTY:
        cls = make_variant(name, bz, basez, roll)
        # base_x_offset=0.0: the 7DOF model was trained before this parameter existed
        env = cls(
            robots=args.robot, controller_configs=ctrl,
            has_renderer=False, has_offscreen_renderer=False,
            use_camera_obs=False, horizon=args.horizon,
            random_order=True, board_pos_noise=0.05,
            base_x_offset=0.0,
        )
        s, pt = run(env, model, args.episodes)
        print(f"{name:16s} {bz:8.2f} {basez:7.3f} {roll:5.0f} "
              f"{s:5d}/{args.episodes:<3d} {pt/args.episodes:12.2f}")
        env.close()


if __name__ == "__main__":
    main()

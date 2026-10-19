"""
OSC_POSE action scale: how much of the command is realised and how fast.
output_max 0.05 m is the impedance controller's TARGET, not a displacement. Measures
steady increment [m/step], time constant (steps to 90 %) and linearity.
X is measured backwards: +X hits the board after a few steps.

    python training/diag_action_scale.py [--table] [--steps N]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import robosuite as suite
from envs.button_board import ButtonBoardEnv

AXES = {0: "X", 1: "Y", 2: "Z"}
SIGNS = {0: -1.0, 1: +1.0, 2: +1.0}
NOMINAL = 0.05


def build_env(robot, horizon):
    ctrl = suite.load_composite_controller_config(controller="BASIC", robot=robot)
    return ButtonBoardEnv(
        robots=robot, controller_configs=ctrl, has_renderer=False,
        has_offscreen_renderer=False, use_camera_obs=False,
        horizon=horizon, random_order=True)


def sweep(env, axis, amplitude, steps, seed):
    """Constant action for `steps`. Returns (deltas (steps,3), collided)."""
    np.random.seed(seed)
    obs = env.reset()
    action = np.zeros(env.action_dim)
    action[axis] = amplitude * SIGNS[axis]

    deltas, collided = [], False
    prev = obs["robot0_eef_pos"].copy()
    for _ in range(steps):
        obs, _, done, _ = env.step(action)
        curr = obs["robot0_eef_pos"]
        deltas.append(curr - prev)
        prev = curr.copy()
        collided |= env._check_board_collision()
        if done:
            break
    return np.array(deltas), collided


def steady_state(d, axis, tail=8):
    return float(np.mean(d[-tail:, axis])) * SIGNS[axis]


def steps_to_90pct(d, axis):
    ss = steady_state(d, axis)
    if ss <= 0:
        return None
    series = d[:, axis] * SIGNS[axis]
    i = int(np.argmax(series >= 0.9 * ss))
    return i if series[i] >= 0.9 * ss else None


def print_table(d):
    print(f"\n  {'step':>4} {'dx':>9} {'dy':>9} {'dz':>9} {'|d|':>9}")
    for t, v in enumerate(d):
        print(f"  {t:4d} {v[0]:+9.5f} {v[1]:+9.5f} {v[2]:+9.5f} "
              f"{np.linalg.norm(v):9.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="Kinova3_6DOF")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    env = build_env(args.robot, horizon=args.steps + 10)
    print(f"{args.robot}, {args.steps} steps, nominal output_max {NOMINAL} m/step")

    print("\n=== amplitude 1.0 ===")
    print(f"{'axis':>4} {'steady':>10} {'% nominal':>10} {'90% @step':>10} "
          f"{'coupling':>9} {'contact':>8}")
    for i in AXES:
        d, hit = sweep(env, i, 1.0, args.steps, seed=1234)
        ss = steady_state(d, i)
        others = [j for j in range(3) if j != i]
        coup = float(np.mean(np.abs(d[-8:, others]))) / ss if ss > 0 else 0.0
        print(f"{AXES[i]:>4} {ss:10.5f} {100*ss/NOMINAL:9.1f}% "
              f"{str(steps_to_90pct(d, i)):>10} {coup:8.1%} {str(hit):>8}")
        if args.table:
            print_table(d)

    print("\n=== linearity ===")
    amps = (0.25, 0.5, 1.0)
    print(f"{'axis':>4} " + " ".join(f"{a:>10}" for a in amps) + f" {'linear':>10}")
    for i in AXES:
        ss = [steady_state(sweep(env, i, a, args.steps, 1234)[0], i) for a in amps]
        ratios = [u / a for u, a in zip(ss, amps)]
        spread = (max(ratios) - min(ratios)) / max(ratios)
        verdict = "yes" if spread < 0.1 else f"NO ({spread:.0%})"
        print(f"{AXES[i]:>4} " + " ".join(f"{u:10.5f}" for u in ss)
              + f" {verdict:>10}")

    # initialization_noise=None: the arm always starts in the same configuration, so this
    # does not test gain vs arm configuration.
    print(f"\n=== repeatability, {args.seeds} seeds (board/base only, same arm pose) ===")
    print(f"{'axis':>4} {'mean':>10} {'min':>10} {'max':>10} {'spread':>8}")
    for i in AXES:
        u = np.array([steady_state(sweep(env, i, 1.0, args.steps, 1234 + s)[0], i)
                      for s in range(args.seeds)])
        print(f"{AXES[i]:>4} {u.mean():10.5f} {u.min():10.5f} {u.max():10.5f} "
              f"{(u.max()-u.min())/u.mean():7.1%}")

    env.close()


if __name__ == "__main__":
    main()

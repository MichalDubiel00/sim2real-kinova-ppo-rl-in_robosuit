"""
Demo wizualizacji srodowisk RL
==============================
Uzycie:
    python eval/demo.py                        # domyslnie: cylinder, 500 krokow
    python eval/demo.py --env button           # srodowisko z guzikiem
    python eval/demo.py --env cylinder --steps 1000 --fps 30
    python eval/demo.py --env button --camera 1 --seed 42
    python eval/demo.py --env cylinder --policy logs/cylinder/best_model/best_model.zip
"""

import argparse
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import robosuite as suite

from envs.cylinder_inspection import CylinderInspectionEnv
from envs.button_press import ButtonPressEnv
from envs.button_board import ButtonBoardEnv

MAX_FPS_DEFAULT = 25


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wizualizacja srodowisk CylinderInspection / ButtonPress"
    )
    parser.add_argument(
        "--env",
        choices=["cylinder", "button", "board"],
        default="cylinder",
        help="Ktore srodowisko uruchomic (domyslnie: cylinder)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Liczba krokow na epizod (domyslnie: 500)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Liczba epizodow (domyslnie: 5)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=MAX_FPS_DEFAULT,
        help=f"Maks. FPS renderowania (domyslnie: {MAX_FPS_DEFAULT})",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help=(
            "ID kamery widoku (domyslnie: 0). "
            "Dostepne: 0=frontview, 1=birdview, 2=agentview, "
            "3=sideview, 4=robot0_robotview, 5=robot0_eye_in_hand"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed losowy (domyslnie: brak — losowy)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Sciezka do zapisanego modelu SB3 (.zip) — jesli podana, uzywa wytrenowanej policy",
    )
    parser.add_argument(
        "--action-noise",
        type=float,
        default=0.3,
        help="Amplituda losowej akcji (random policy), domyslnie: 0.3",
    )
    return parser.parse_args()


def load_policy(path):
    """Laduje policy SB3 (PPO) z pliku .zip."""
    from stable_baselines3 import PPO
    print(f"Ladowanie policy: {path}")
    return PPO.load(path)


def flatten_obs(obs_dict):
    return np.concatenate(
        [np.asarray(v).flatten() for v in obs_dict.values()],
        dtype=np.float32,
    )


def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Cylinder env uzywa SpotWithArm, pozostale uzywaja Kinova3
    robot_name = "SpotWithArm" if args.env == "cylinder" else "Kinova3"
    cfg = suite.load_composite_controller_config(controller="BASIC", robot=robot_name)

    env_kwargs = dict(
        robots=robot_name,
        controller_configs=cfg,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        render_visual_mesh=True,
        render_collision_mesh=False,
        control_freq=20,
        horizon=args.steps,
    )

    env_display = {"cylinder": "CylinderInspection", "button": "ButtonPress", "board": "ButtonBoard"}
    print(f"\nUruchamianie: {env_display[args.env]}")
    print(f"Epizody: {args.episodes}  |  Kroki/epizod: {args.steps}  |  FPS: {args.fps}")
    if args.policy:
        print(f"Policy: {args.policy}")
    else:
        print(f"Policy: losowa (szum={args.action_noise})")

    if args.env == "cylinder":
        env = CylinderInspectionEnv(**env_kwargs)
    elif args.env == "button":
        env = ButtonPressEnv(**env_kwargs)
    else:
        env = ButtonBoardEnv(**env_kwargs)

    policy = load_policy(args.policy) if args.policy else None

    env.reset()
    env.render()  # initialise viewer before setting camera
    if args.env == "board" and args.camera == 0:
        cam_id = env.sim.model.camera_name2id("board_view")
        env.viewer.set_camera(camera_id=cam_id)
    else:
        env.viewer.set_camera(camera_id=args.camera)

    for ep in range(args.episodes):
        obs_dict = env.reset()
        ep_reward = 0.0
        success = False

        print(f"\n--- Epizod {ep + 1}/{args.episodes} ---")

        for step in range(args.steps):
            t_start = time.time()

            if policy is not None:
                obs_flat = flatten_obs(obs_dict)
                action, _ = policy.predict(obs_flat, deterministic=True)
            else:
                action = np.random.uniform(
                    -args.action_noise, args.action_noise, env.action_dim
                )

            obs_dict, reward, done, info = env.step(action)
            env.render()
            ep_reward += reward

            if env._check_success():
                success = True

            # Ogranicz FPS
            elapsed = time.time() - t_start
            wait = 1.0 / args.fps - elapsed
            if wait > 0:
                time.sleep(wait)

            if done:
                break

        status = "SUKCES" if success else "brak sukcesu"
        print(f"Reward: {ep_reward:.3f}  |  {status}  |  Krokow: {step + 1}")

    env.close()
    print("\nDemo zakonczone.")


if __name__ == "__main__":
    main()

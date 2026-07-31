"""
Wizualizacja ButtonBoard-v0
===========================
Uzycie:
    python training/watch_board.py                        # okno na zywo, auto-model
    python training/watch_board.py --model logs/board/checkpoints/ppo_board_500000_steps.zip
    python training/watch_board.py --random               # losowe akcje (bez modelu)
    python training/watch_board.py --record out.mp4       # bez okna, zapis wideo (dziala tez przez SSH)
    python training/watch_board.py --episodes 5

UWAGA: import stable_baselines3 ciagnie przez tensorboard TensorFlow,
a TF w tym srodowisku psuje pozniejszy lazy-import torch._dynamo
(segfault w PPO.load przy tworzeniu optymalizatora Adam). TF nie jest
do niczego potrzebny, wiec blokujemy jego import ponizej.
"""

import argparse
import glob
import os
import re
import sys

sys.modules["tensorflow"] = None  # blokada: tensorboard uzyje wlasnego stuba

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def flatten_obs(obs_dict):
    return np.concatenate(
        [np.asarray(v).flatten() for v in obs_dict.values()],
        dtype=np.float32,
    )


def find_model():
    """Najpierw best_model, potem checkpoint z najwieksza liczba krokow."""
    best = "logs/board/best_model/best_model.zip"
    if os.path.exists(best):
        return best
    ckpts = glob.glob("logs/board/checkpoints/ppo_board_*_steps.zip")
    if ckpts:
        return max(ckpts, key=lambda p: int(re.search(r"_(\d+)_steps", p).group(1)))
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--random", action="store_true",
                        help="losowe akcje zamiast modelu")
    parser.add_argument("--record", type=str, default=None, metavar="PLIK.mp4",
                        help="zapis wideo offscreen zamiast okna (dziala bez ekranu)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--camera", type=str, default="board_view")
    parser.add_argument("--robot", type=str, default="Kinova3",
                        help="Kinova3 (7DOF, stock) albo Kinova3_6DOF "
                             "(uczelniany; modele 7DOF NIE dzialaja — inny "
                             "wymiar obserwacji)")
    parser.add_argument("--activation-depth", type=float, default=None,
                        help="prog wcisniecia [m]; domyslnie 0.012 (7DOF), "
                             "modele 6DOF trenowane na 0.010")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = None
    if not args.random:
        model_path = args.model or find_model()
        if model_path is None:
            print("Brak modelu (jeszcze zaden checkpoint nie istnieje) — losowe akcje.")

    if args.record:
        os.environ.setdefault("MUJOCO_GL", "egl")

    # 1) Najpierw srodowisko + viewer (patrz UWAGA w naglowku)
    import robosuite as suite
    from envs.button_board import ButtonBoardEnv

    controller_cfg = suite.load_composite_controller_config(
        controller="BASIC", robot=args.robot)

    env = ButtonBoardEnv(
        robots=args.robot,
        controller_configs=controller_cfg,
        has_renderer=not args.record,
        render_camera=args.camera,
        has_offscreen_renderer=bool(args.record),
        use_camera_obs=False,
        horizon=1000,
        random_order=True,
        board_pos_noise=0.05,
        activation_depth=args.activation_depth,
    )
    env.reset()
    if not args.record:
        env.render()  # otworz okno zanim zaladujemy torcha

    # 2) Dopiero teraz stable_baselines3 / torch
    model = None
    if model_path is not None:
        from stable_baselines3 import PPO
        print(f"Laduje model: {model_path}")
        model = PPO.load(model_path, device="cpu")

    frames = []
    for ep in range(args.episodes):
        obs = env.reset()
        total, done, t = 0.0, False, 0
        while not done:
            if model is not None:
                action, _ = model.predict(flatten_obs(obs), deterministic=True)
            else:
                action = np.random.uniform(-0.3, 0.3, env.action_dim)
            obs, reward, done, info = env.step(action)
            total += reward
            t += 1
            if args.record:
                if t % 2 == 0:  # 10 fps przy control_freq 20
                    frames.append(env.sim.render(
                        camera_name=args.camera, width=640, height=480)[::-1])
            else:
                env.render()
        print(f"Epizod {ep}: krokow={t}, reward={total:.3f}, "
              f"wcisniete={env._pressed}, sukces={env._check_success()}")

    env.close()

    if args.record and frames:
        import subprocess
        tmp_dir = "/tmp/watch_board_frames"
        os.makedirs(tmp_dir, exist_ok=True)
        from PIL import Image
        for i, f in enumerate(frames):
            Image.fromarray(f).save(f"{tmp_dir}/{i:06d}.png")
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", "10",
             "-i", f"{tmp_dir}/%06d.png",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", args.record],
            check=True, capture_output=True)
        for p in glob.glob(f"{tmp_dir}/*.png"):
            os.remove(p)
        print(f"Wideo zapisane: {args.record}")


if __name__ == "__main__":
    main()

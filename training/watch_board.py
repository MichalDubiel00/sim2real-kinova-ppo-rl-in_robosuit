"""
Wizualizacja ButtonBoard-v0
===========================
Uzycie:
    python training/watch_board.py                        # okno na zywo, auto-model
    python training/watch_board.py --model logs/board/checkpoints/ppo_board_500000_steps.zip
    python training/watch_board.py --random               # losowe akcje (bez modelu)
    python training/watch_board.py --record out.mp4       # bez okna, zapis wideo (dziala tez przez SSH)
    python training/watch_board.py --episodes 5

Nagranie modelu na sprzet w OBWIEDNI SPRZETOWEJ (te same warunki co ewaluacja
n=100 z docs 6.40 — bez tych flag wideo pokazuje INNE zadanie):
    python training/watch_board.py --record film.mp4 \
        --model logs/board_6dof/kinova_ft_lr2/checkpoints/ppo_board_28023552_steps.zip \
        --robot Kinova3_6DOF --horizon 2000 --kinova-limits \
        --estop-config training/configs/ppo_board_6dof_estop_hw.yaml \
        --seeds 1262,1253,1258 --fps 20 --width 1280 --height 720

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
    parser.add_argument("--horizon", type=int, default=1000,
                        help="dlugosc epizodu; modele 6DOF etapu 1 trenowane "
                             "na 2000")
    parser.add_argument("--free-gripper", action="store_true",
                        help="chwytak sterowany przez polityke "
                             "(force_gripper_closed=False); musi byc zgodne "
                             "z trybem treningu modelu — patrz docs 6.15")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed bazowy epizodow (konwencja jak w "
                             "compare_models.py: epizod k dostaje seed+k); "
                             "pozwala odtworzyc konkretny epizod z ewaluacji")
    parser.add_argument("--seeds", type=str, default=None,
                        help="lista seedow po przecinku (np. 1262,1253,1258) "
                             "zamiast --seed/--episodes; nagrywa dokladnie te "
                             "epizody, w tej kolejnosci")
    # --- obwiednia sprzetowa: te same flagi co compare_models.py ---
    parser.add_argument("--kinova-limits", action="store_true",
                        help="limity wg instrukcji Gen3 R07: stawy "
                             "[1.39 x3, 1.22 x3] w trybie 'scale' + limity "
                             "twist (0.5 m/s, 0.8727 rad/s)")
    parser.add_argument("--hw-limits", action="store_true",
                        help="limit 0.8727 rad/s na kazdy staw, tryb 'clip' "
                             "(ros2_kortex joint_limits.yaml)")
    parser.add_argument("--estop", action="store_true",
                        help="profil E-stopu ESTOP_PROFILE_EXAMPLE")
    parser.add_argument("--estop-config", type=str, default=None,
                        help="YAML z kluczem button_estop_profile (i opcjonalnie "
                             "activation_depth), np. configs/ppo_board_6dof_estop_hw.yaml")
    parser.add_argument("--button-stiffness", type=float, default=None,
                        help="sztywnosc przycisku [N/m]; None = model domyslny")
    parser.add_argument("--pos-max", type=float, default=None,
                        help="output_max translacji OSC [m/krok]")
    parser.add_argument("--rot-max", type=float, default=None,
                        help="output_max obrotu OSC [rad/krok]")
    parser.add_argument("--acc-limits", action="store_true",
                        help="twarde limity przyspieszen Gen3 wg instrukcji R07 tab. 48: "
                             "[5.2 x3, 10.0 x3] rad/s^2 (docs 6.44). BEZ tego nagranie pokazuje "
                             "ruchy, ktorych prawdziwy robot NIE WYKONA")
    parser.add_argument("--base-x-offset", type=float, default=None, metavar="M",
                        help="odleglosc postoju Spota (domyslne env -0.10; biegi od 6.42: -0.34)")
    parser.add_argument("--init-qpos", type=str, default=None, metavar="q1,..,q6",
                        help="poza startowa, 6 katow po przecinku")
    parser.add_argument("--spot-body", action="store_true",
                        help="bryla korpusu Spota pod ramieniem (zmierzona z Webotsa, docs 6.41)")
    parser.add_argument("--spot-front-cam", action="store_true",
                        help="przednia kamera Spota (obrys Kinecta jako bufor)")
    parser.add_argument("--no-finger-gate", action="store_true",
                        help="wylacza wymog kontaktu palcow przy wcisnieciu")
    # --- parametry nagrania ---
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=10,
                        help="klatek/s w pliku; przy control_freq 20 Hz wartosc "
                             "20 daje czas rzeczywisty, 10 = co druga klatka")
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
    from envs.controller_scale import with_osc_output_max
    # stale limitow trzymane w JEDNYM miejscu (compare_models.py), zeby wideo
    # i pomiar nie rozjechaly sie po kolejnej korekcie limitow
    from compare_models import HW_JOINT_VEL, KINOVA_JOINT_VEL, KINOVA_TWIST

    estop_prof = ButtonBoardEnv.ESTOP_PROFILE_EXAMPLE if args.estop else None
    if args.estop_config:
        if args.estop:
            raise SystemExit("--estop i --estop-config wykluczaja sie")
        import yaml
        _c = yaml.safe_load(open(args.estop_config))
        estop_prof = _c["button_estop_profile"]
        if args.activation_depth is None:
            args.activation_depth = _c.get("activation_depth")
        print(f"[estop] z {args.estop_config}: {estop_prof}, prog {args.activation_depth}")

    controller_cfg = with_osc_output_max(
        suite.load_composite_controller_config(
            controller="BASIC", robot=args.robot),
        args.pos_max, args.rot_max)

    env = ButtonBoardEnv(
        robots=args.robot,
        controller_configs=controller_cfg,
        has_renderer=not args.record,
        render_camera=args.camera,
        has_offscreen_renderer=bool(args.record),
        use_camera_obs=False,
        horizon=args.horizon,
        random_order=True,
        # board_pos_noise / base_z_noise / percep_* CELOWO nie podawane -
        # biora domyslne z ButtonBoardEnv (patrz compare_models.py).
        activation_depth=args.activation_depth,
        force_gripper_closed=not args.free_gripper,
        finger_press_gate=not args.no_finger_gate,
        joint_vel_limits=(KINOVA_JOINT_VEL if args.kinova_limits
                          else HW_JOINT_VEL if args.hw_limits else None),
        joint_vel_limit_mode=("scale" if args.kinova_limits else "clip"),
        eef_twist_limits=(KINOVA_TWIST if args.kinova_limits else None),
        **({"base_x_offset": args.base_x_offset} if args.base_x_offset is not None else {}),
        init_qpos=([float(x) for x in args.init_qpos.split(",")] if args.init_qpos else None),
        joint_acc_limits=([5.2, 5.2, 5.2, 10.0, 10.0, 10.0] if args.acc_limits else None),
        spot_body=args.spot_body,
        spot_front_cam=args.spot_front_cam,
        button_stiffness=args.button_stiffness,
        button_estop_profile=estop_prof,
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

    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    elif args.seed is not None:
        seeds = [args.seed + ep for ep in range(args.episodes)]
    else:
        seeds = [None] * args.episodes

    frame_every = max(1, round(20 / max(args.fps, 1)))  # control_freq 20 Hz

    frames = []
    for ep, seed in enumerate(seeds):
        if seed is not None:
            # ta sama konwencja parowania co compare_models.py: epizod o tym
            # samym seedzie ma identyczne warunki, wiec mozna odtworzyc
            # konkretny (np. udany) epizod z CSV ewaluacji
            np.random.seed(seed)
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
                if t % frame_every == 0:
                    frames.append(env.sim.render(
                        camera_name=args.camera,
                        width=args.width, height=args.height)[::-1])
            else:
                env.render()
        print(f"Epizod {ep} (seed={seed}): krokow={t}, reward={total:.3f}, "
              f"wcisniete={env._pressed}, sukces={env._check_success()}")

    env.close()

    if args.record and frames:
        import subprocess
        # Katalog UNIKALNY dla procesu — stala sciezka powodowala, ze
        # rownolegle uruchomione nagrania nadpisywaly sobie klatki
        # i ffmpeg sklejal film z kilku biegow (incydent 2026-09-07).
        tmp_dir = f"/tmp/watch_board_frames_{os.getpid()}"
        os.makedirs(tmp_dir, exist_ok=True)
        from PIL import Image
        for i, f in enumerate(frames):
            Image.fromarray(f).save(f"{tmp_dir}/{i:06d}.png")
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(args.fps),
             "-i", f"{tmp_dir}/%06d.png",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", args.record],
            check=True, capture_output=True)
        for p in glob.glob(f"{tmp_dir}/*.png"):
            os.remove(p)
        os.rmdir(tmp_dir)
        print(f"Wideo zapisane: {args.record}")


if __name__ == "__main__":
    main()

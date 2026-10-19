"""
ButtonBoard-v0 viewer / recorder.
    python training/watch_board.py                        # live window, auto model
    python training/watch_board.py --random               # random actions
    python training/watch_board.py --record out.mp4       # offscreen video (works over SSH)

For hardware-envelope recordings pass the same flags as the evaluation, e.g.:
    python training/watch_board.py --record film.mp4 --model <ckpt.zip> \
        --robot Kinova3_6DOF --horizon 2000 --kinova-limits \
        --estop-config training/configs/ppo_board_6dof_estop_hw.yaml --seeds 1262,1253
"""

import argparse
import glob
import os
import re
import sys

sys.modules["tensorflow"] = None  # tensorboard falls back to its own stub (docs 7.6)

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def flatten_obs(obs_dict):
    return np.concatenate(
        [np.asarray(v).flatten() for v in obs_dict.values()],
        dtype=np.float32,
    )


def find_model():
    """best_model first, else the checkpoint with the most steps."""
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
                        help="random actions instead of a model")
    parser.add_argument("--record", type=str, default=None, metavar="FILE.mp4",
                        help="record offscreen video instead of a window (works headless)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--camera", type=str, default="board_view")
    parser.add_argument("--robot", type=str, default="Kinova3",
                        help="Kinova3 (7DOF, stock) or Kinova3_6DOF "
                             "(lab variant; 7DOF models do NOT work - different "
                             "observation size)")
    parser.add_argument("--activation-depth", type=float, default=None,
                        help="press threshold [m]; default 0.012 (7DOF), "
                             "6DOF models were trained at 0.010")
    parser.add_argument("--horizon", type=int, default=1000,
                        help="episode length; stage-1 6DOF models were trained "
                             "at 2000")
    parser.add_argument("--free-gripper", action="store_true",
                        help="gripper controlled by the policy "
                             "(force_gripper_closed=False); must match "
                             "the model's training mode - see docs 6.15")
    parser.add_argument("--seed", type=int, default=None,
                        help="base seed (same convention as "
                             "compare_models.py: episode k gets seed+k); "
                             "lets you replay a specific evaluation episode")
    parser.add_argument("--seeds", type=str, default=None,
                        help="comma-separated seeds (e.g. 1262,1253,1258) "
                             "instead of --seed/--episodes; records exactly these "
                             "episodes, in this order")
    # --- hardware envelope: same flags as compare_models.py ---
    parser.add_argument("--kinova-limits", action="store_true",
                        help="limits per Gen3 R07 manual: joints "
                             "[1.39 x3, 1.22 x3] in 'scale' mode + limits "
                             "twist (0.5 m/s, 0.8727 rad/s)")
    parser.add_argument("--hw-limits", action="store_true",
                        help="0.8727 rad/s limit on every joint, 'clip' mode "
                             "(ros2_kortex joint_limits.yaml)")
    parser.add_argument("--estop", action="store_true",
                        help="E-stop profile ESTOP_PROFILE_EXAMPLE")
    parser.add_argument("--estop-config", type=str, default=None,
                        help="YAML with a button_estop_profile key (and optionally "
                             "activation_depth), np. configs/ppo_board_6dof_estop_hw.yaml")
    parser.add_argument("--button-stiffness", type=float, default=None,
                        help="button stiffness [N/m]; None = default model")
    parser.add_argument("--pos-max", type=float, default=None,
                        help="OSC translation output_max [m/step]")
    parser.add_argument("--rot-max", type=float, default=None,
                        help="OSC rotation output_max [rad/step]")
    parser.add_argument("--acc-limits", action="store_true",
                        help="hard Gen3 acceleration limits, R07 manual table 48: "
                             "[5.2 x3, 10.0 x3] rad/s^2 (docs 6.44). WITHOUT it the video shows "
                             "motions the real robot will NOT perform")
    parser.add_argument("--acc-mode", choices=["clip", "scale"], default="clip",
                        help="how to clip accelerations: clip = per joint, scale = "
                             "proportional (firmware twist mode; docs 6.50). "
                             "arm_* models were TRAINED and EVALUATED with scale - recording with clip "
                             "shows a DIFFERENT task (collisions 7.0 vs 14.7)")
    parser.add_argument("--action-delay", type=int, default=None, metavar="K",
                        help="loop dead time: the command from step t acts at t+K (docs 6.52). "
                             "1 step = 50 ms. arm_d was trained with random [0,2]")
    parser.add_argument("--base-x-offset", type=float, default=None, metavar="M",
                        help="Spot stand-off (env default -0.10; runs since 6.42: -0.34)")
    parser.add_argument("--init-qpos", type=str, default=None, metavar="q1,..,q6",
                        help="start pose, 6 comma-separated angles")
    parser.add_argument("--spot-body", action="store_true",
                        help="Spot body under the arm (measured in Webots, docs 6.41)")
    parser.add_argument("--spot-front-cam", action="store_true",
                        help="front Spot camera (Kinect outline as a buffer)")
    parser.add_argument("--no-finger-gate", action="store_true",
                        help="disables the finger-contact requirement for a press")
    # --- recording ---
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=10,
                        help="frames/s in the file; at control_freq 20 Hz "
                             "20 gives real time, 10 = every other frame")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = None
    if not args.random:
        model_path = args.model or find_model()
        if model_path is None:
            print("No model (no checkpoint exists yet) - random actions.")

    if args.record:
        os.environ.setdefault("MUJOCO_GL", "egl")

    # 1) Env + viewer first, before torch
    import robosuite as suite
    from envs.button_board import ButtonBoardEnv
    from envs.controller_scale import with_osc_output_max
    # limit constants live in compare_models.py only
    from compare_models import HW_JOINT_VEL, KINOVA_JOINT_VEL, KINOVA_TWIST

    estop_prof = ButtonBoardEnv.ESTOP_PROFILE_EXAMPLE if args.estop else None
    if args.estop_config:
        if args.estop:
            raise SystemExit("--estop and --estop-config are mutually exclusive")
        import yaml
        _c = yaml.safe_load(open(args.estop_config))
        estop_prof = _c["button_estop_profile"]
        if args.activation_depth is None:
            args.activation_depth = _c.get("activation_depth")
        print(f"[estop] from {args.estop_config}: {estop_prof}, threshold {args.activation_depth}")

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
        # noise params deliberately omitted: env defaults, as in compare_models.py
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
        joint_acc_limit_mode=args.acc_mode,
        action_delay_steps=args.action_delay,
        spot_body=args.spot_body,
        spot_front_cam=args.spot_front_cam,
        button_stiffness=args.button_stiffness,
        button_estop_profile=estop_prof,
    )
    env.reset()
    if not args.record:
        env.render()  # open the window before loading torch

    # 2) Only now stable_baselines3 / torch
    model = None
    if model_path is not None:
        from stable_baselines3 import PPO
        print(f"Loading model: {model_path}")
        model = PPO.load(model_path, device="cpu")

    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    elif args.seed is not None:
        seeds = [args.seed + ep for ep in range(args.episodes)]
    else:
        seeds = [None] * args.episodes

    frame_every = max(1, round(20 / max(args.fps, 1)))  # control_freq 20 Hz

    # Frames go straight to disk (keeping them in memory OOMed at 11 GB, 2026-10-10).
    tmp_dir = n_frames = None
    if args.record:
        import subprocess
        from PIL import Image
        # Per-process directory: parallel recordings overwrote each other's frames.
        tmp_dir = f"/tmp/watch_board_frames_{os.getpid()}"
        os.makedirs(tmp_dir, exist_ok=True)
        n_frames = 0
    for ep, seed in enumerate(seeds):
        if seed is not None:
            # same seeding as compare_models.py, so a CSV episode can be replayed
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
                    Image.fromarray(env.sim.render(
                        camera_name=args.camera,
                        width=args.width, height=args.height)[::-1]
                    ).save(f"{tmp_dir}/{n_frames:06d}.png")
                    n_frames += 1
            else:
                env.render()
        print(f"Episode {ep} (seed={seed}): steps={t}, reward={total:.3f}, "
              f"pressed={env._pressed}, success={env._check_success()}")

    env.close()

    if args.record and n_frames:
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(args.fps),
             "-i", f"{tmp_dir}/%06d.png",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", args.record],
            check=True, capture_output=True)
        for p in glob.glob(f"{tmp_dir}/*.png"):
            os.remove(p)
        os.rmdir(tmp_dir)
        print(f"Video saved: {args.record}")


if __name__ == "__main__":
    main()

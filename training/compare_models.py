"""
Paired model comparison on ButtonBoard-v0.
    python training/compare_models.py A=path1.zip B=path2.zip --episodes 100
    python training/compare_models.py model.zip --episodes 50 --csv results.csv

Pairing: np.random is re-seeded with (seed + episode) before every episode, so
episode k has identical conditions (board pose, order, perception noise sequence)
for every model; trajectories differ only by the policy's actions.

Metrics: success (all 5 pressed), steps, collision steps (board), misuse steps
(arm link on a button), smoothness mean ||a_t - a_{t-1}||, stuck (EEF bounding box
of the last 200 steps < 3 cm).
"""

import argparse
import csv
import os
import sys

sys.modules["tensorflow"] = None  # docs 7.6 (TF vs torch._dynamo segfault)

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import robosuite as suite
from envs.button_board import ButtonBoardEnv
from envs.controller_scale import with_osc_output_max

# Gen3 6DOF joint velocity limits: 0.8727 rad/s on all joints (kortex_description
# joint_limits.yaml, matches the datasheet; not the PR #241 values, docs 6.31).
HW_JOINT_VEL = [0.8727] * 6
# Gen3 R07 manual (docs 6.37): general limits table 46; twist mode scales proportionally
KINOVA_JOINT_VEL = [1.39, 1.39, 1.39, 1.22, 1.22, 1.22]
KINOVA_TWIST = (0.5, 0.8727)
# Gen3 R07 table 48: HARD acceleration limits (always apply). Table 49 covers only
# angular joystick and joint trajectory, not twist (docs 6.44).
HW_JOINT_ACC = [5.2, 5.2, 5.2, 10.0, 10.0, 10.0]


def flatten_obs(obs_dict):
    return np.concatenate(
        [np.asarray(v).flatten() for v in obs_dict.values()],
        dtype=np.float32,
    )


def run_episode(env, model, seed, action_ema=None):
    """One episode with seed-defined conditions; returns metrics.
    action_ema: low-pass alpha on the action (None/1 = off), a deployment-side filter."""
    np.random.seed(seed)  # pairing: identical conditions for every model
    obs = env.reset()

    arm = env.robots[0].arms[0]
    eef_site = env.robots[0].eef_site_id[arm]

    done, t, coll_steps, misuse_steps, rate_sum = False, 0, 0, 0, 0.0
    spot_steps = 0
    prev_a = None
    eef_hist = []

    a_filt = None
    while not done:
        action, _ = model.predict(flatten_obs(obs), deterministic=True)
        if action_ema is not None and action_ema < 1.0:
            a_filt = (action.astype(np.float64) if a_filt is None
                      else action_ema * action + (1.0 - action_ema) * a_filt)
            action = a_filt.astype(np.float32)
        obs, reward, done, _ = env.step(action)
        t += 1
        if env._check_board_collision():
            coll_steps += 1
        if env._check_button_misuse():
            misuse_steps += 1
        if env._check_spot_collision():
            spot_steps += 1
        if prev_a is not None:
            rate_sum += float(np.linalg.norm(action - prev_a))
        prev_a = action
        eef_hist.append(env.sim.data.site_xpos[eef_site].copy())

    success = env._check_success()
    stuck = False
    if not success and len(eef_hist) >= 200:
        seg = np.array(eef_hist[-200:])
        stuck = float(np.linalg.norm(seg.max(axis=0) - seg.min(axis=0))) < 0.03

    return {
        "success": bool(success),
        "pressed": int(sum(env._pressed)),
        "pressed_mask": [bool(p) for p in env._pressed],
        "steps": t,
        "coll_steps": coll_steps,
        "spot_steps": spot_steps,
        "misuse_steps": misuse_steps,
        "action_rate": rate_sum / max(t - 1, 1),
        "stuck": bool(stuck),
    }


def summarize(label, rows):
    succ = np.array([r["success"] for r in rows])
    steps = np.array([r["steps"] for r in rows])
    coll = np.array([r["coll_steps"] for r in rows])
    misuse = np.array([r["misuse_steps"] for r in rows])
    rate = np.array([r["action_rate"] for r in rows])
    pressed = np.array([r["pressed"] for r in rows])
    stuck = np.array([r["stuck"] for r in rows])

    n = len(rows)
    print(f"\n=== {label} (n={n}) ===")
    print(f"successes:                {succ.sum()}/{n} ({100*succ.mean():.0f}%)")
    if succ.any():
        print(f"mean time to success:     {steps[succ].mean():.0f} steps "
              f"({steps[succ].mean()/20:.1f} s at 20 Hz)")
    print(f"episodes with collision:  {(coll > 0).sum()}/{n}")
    print(f"mean collision steps/ep:  {coll.mean():.1f}")
    spot = np.array([r.get("spot_steps", 0) for r in rows])
    if spot.sum() or any("spot_steps" in r for r in rows):
        print(f"episodes with Spot contact: {(spot > 0).sum()}/{n}")
        print(f"mean Spot contact steps:  {spot.mean():.1f}")
    print(f"episodes with button misuse: {(misuse > 0).sum()}/{n}")
    print(f"mean misuse steps/ep:     {misuse.mean():.1f}")
    print(f"smoothness mean ||da||/step: {rate.mean():.4f}")
    fails = [int(((~succ) & (pressed == k)).sum()) for k in range(5)]
    print(f"failures by pressed count (0..4/5): {fails}, stuck: {stuck.sum()}")
    per_btn = np.array([r["pressed_mask"] for r in rows])  # (n, 5)
    print("presses per button:       "
          + ", ".join(f"btn{i}: {per_btn[:, i].sum()}/{n}" for i in range(per_btn.shape[1])))
    return succ


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+",
                   help="model paths, optionally 'label=path.zip'")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--button-stiffness", type=float, default=None,
                   help="cap spring stiffness [N/m]; default 500 from button.xml (~6 N "
                        "at 12 mm). Typical E-stop 20-40 N -> 1700-3300 N/m (docs 6.33)")
    p.add_argument("--pos-max", type=float, default=None,
                   help="OSC translation output_max [m/step]; MUST match the model's "
                        "training (default 0.05; docs 6.36)")
    p.add_argument("--rot-max", type=float, default=None,
                   help="OSC rotation output_max [rad/step]; MUST match the model's "
                        "training (default 0.5; docs 6.35)")
    p.add_argument("--estop-config", type=str, default=None,
                   help="training YAML: takes button_estop_profile and "
                        "activation_depth from it (e.g. E-stop 20 N / 6 mm). Excludes --estop")
    p.add_argument("--estop", action="store_true",
                   help="non-linear latching E-stop model "
                        "ButtonBoardEnv.ESTOP_PROFILE_EXAMPLE (15->35 N up to 3.5 mm, "
                        "drop to 5 N, 5 mm travel, latch; docs 6.34). Without "
                        "--activation-depth the threshold = contact opening point (3.8 mm)")
    p.add_argument("--kinova-limits", action="store_true",
                   help="Gen3 R07 limits for twist mode: joints 1.39/1.22 rad/s "
                        "scaled proportionally + tool 0.5 m/s, 0.8727 rad/s (docs 6.37)")
    p.add_argument("--hw-limits", action="store_true",
                   help="apply Gen3 firmware joint velocity limits "
                        "(ros2_kortex joint_limits.yaml): 0.8727 rad/s on all joints")
    p.add_argument("--gain", type=float, default=None,
                   help="FIXED mechanism gain on all 6 axes "
                        "(1.0 = as in training). Simulates the sim2sim mismatch "
                        "measured in Webots: 0.32 translation, 0.10 rotation (docs 6.30).")
    p.add_argument("--robot", type=str, default="Kinova3",
                   help="Kinova3 (7DOF, stock) or Kinova3_6DOF (lab variant); "
                        "6DOF models are NOT compatible with 7DOF (obs 120 vs 128)")
    p.add_argument("--activation-depth", type=float, default=None,
                   help="press threshold in metres; default from the env (0.012), "
                        "stage-1 6DOF models were trained at 0.010")
    p.add_argument("--no-finger-gate", action="store_true",
                   help="disable the finger gate. MUST match the mode the model was "
                        "trained in - curriculum stage 1 (`depress`, `hwlim_s1`) uses "
                        "finger_press_gate=false, so evaluating it with the gate on "
                        "gives 0/40 despite reaching the button")
    p.add_argument("--free-gripper", action="store_true",
                   help="gripper controlled by the policy (force_gripper_closed"
                        "=False). MUST match the mode the model was "
                        "trained in - a mismatch changes the task (A/B: "
                        "82%% -> 50%% on the same model, docs 6.15)")
    p.add_argument("--acc-limits", action="store_true",
                   help="HARD Gen3 6DOF acceleration limits, R07 manual table 48: "
                        "[5.2 x3, 10.0 x3] rad/s^2. NOTE: 1.0 rad/s^2 from table 49 applies "
                        "ONLY to angular joystick/joint trajectory, NOT to twist (docs 6.44)")
    p.add_argument("--action-delay", type=int, default=None, metavar="K",
                   help="loop dead time: the command from step t acts at t+K "
                        "(docs 6.52). robosuite has ZERO dead time; on hardware "
                        "every ROS->firmware->ROS hop adds some. 1 step = 50 ms")
    p.add_argument("--acc-mode", choices=["clip", "scale"], default="clip",
                   help="acceleration clipping semantics (docs 6.50). clip = independently "
                        "per joint (changes motion direction). scale = the whole velocity "
                        "increment scaled by one factor, direction kept - "
                        "what the Gen3 firmware does in twist mode (R07 manual). Both "
                        "keep the SAME envelope; they differ in direction after clipping")
    p.add_argument("--acc-ref", choices=["qvel", "cmd"], default="qvel",
                   help="reference of the acceleration limit (docs 6.48). "
                        "qvel = relative to the measured velocity of the previous substep "
                        "(legacy; the budget is eaten by OSC ripple, median "
                        "40 rad/s^2 on j3 with the arm still at a=0). cmd = relative to "
                        "the own previous command, i.e. the slope of the COMMANDED velocity "
                        "profile - what the firmware limits")
    p.add_argument("--base-x-offset", type=float, default=None, metavar="M",
                   help="Spot stand-off: arm base shift along X. "
                        "Env default is -0.10; runs since docs 6.42 use -0.34. "
                        "WITHOUT it a model trained at -0.34 would be evaluated in a DIFFERENT scene")
    p.add_argument("--init-qpos", type=str, default=None, metavar="q1,..,q6",
                   help="start pose as 6 comma-separated angles; default = robot default "
                        "(gripper pointing down, the wrist camera cannot see the board)")
    p.add_argument("--obs-acc-saturation", action="store_true",
                   help="add the saturation signal to the observation (obs 120 -> 132, docs 6.47). Models before 2026-10-07 have input 120 and will NOT load with this flag")
    p.add_argument("--action-rate-limit", type=float, default=None, metavar="D",
                   help="command rate LIMIT: |a_t - a_{t-1}| <= D per component, "
                        "applied before the controller (docs 6.47). A constraint, not a penalty")
    p.add_argument("--approach-speed-cost", type=float, default=0.0,
                   help="penalty on approach speed above v_ok = distance/0.3 s (docs 6.45); "
                        "usually 0 in EVALUATION - changes the reward, not the behaviour")
    p.add_argument("--spot-body", action="store_true",
                   help="insert the Spot body under the arm (measured in Webots, "
                        "docs 6.41). Absent in older training - enabling it changes the task")
    p.add_argument("--spot-front-cam", action="store_true",
                   help="insert the front Spot camera (Kinect outline from Webots as a buffer): "
                        "28 cm wide bar, 21-27 cm in front of the base, at its height")
    p.add_argument("--action-ema", type=float, default=None, metavar="ALPHA",
                   help="low-pass filter on the action: a_t = alpha*a + (1-alpha)*a_{t-1}. "
                        "1.0 or unset = no filter. A deployment-side change - "
                        "softens bang-bang (||da|| 1.335/step) before hardware")
    p.add_argument("--csv", type=str, default=None,
                   help="write per-episode metrics to a CSV file")
    return p.parse_args()


def main():
    args = parse_args()
    from stable_baselines3 import PPO

    specs = []
    for m in args.models:
        label, _, path = m.rpartition("=")
        specs.append((label or os.path.basename(path), path or m))

    if args.init_qpos:
        args.init_qpos = [float(x) for x in args.init_qpos.split(",")]
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
    ctrl = with_osc_output_max(suite.load_composite_controller_config(
        controller="BASIC", robot=args.robot), args.pos_max, args.rot_max)
    env = ButtonBoardEnv(
        robots=args.robot, controller_configs=ctrl,
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, horizon=args.horizon,
        # noise params deliberately omitted: env defaults, same as training (docs 6.24).
        random_order=True,
        activation_depth=args.activation_depth,
        force_gripper_closed=not args.free_gripper,
        finger_press_gate=not args.no_finger_gate,
        # degenerate range = FIXED gain
        gain_noise_lin=((args.gain, args.gain) if args.gain is not None else None),
        gain_noise_rot=((args.gain, args.gain) if args.gain is not None else None),
        joint_acc_limits=(HW_JOINT_ACC if args.acc_limits else None),
        joint_acc_limit_ref=args.acc_ref,
        joint_acc_limit_mode=args.acc_mode,
        action_delay_steps=args.action_delay,
        joint_vel_limits=(KINOVA_JOINT_VEL if args.kinova_limits
                          else HW_JOINT_VEL if args.hw_limits else None),
        joint_vel_limit_mode=("scale" if args.kinova_limits else "clip"),
        eef_twist_limits=(KINOVA_TWIST if args.kinova_limits else None),
        **({"base_x_offset": args.base_x_offset} if args.base_x_offset is not None else {}),
        init_qpos=args.init_qpos,
        obs_acc_saturation=args.obs_acc_saturation,
        action_rate_limit=args.action_rate_limit,
        approach_speed_cost=args.approach_speed_cost,
        spot_body=args.spot_body,
        spot_front_cam=args.spot_front_cam,
        button_stiffness=args.button_stiffness,
        button_estop_profile=estop_prof,
    )

    all_rows = {}
    for label, path in specs:
        print(f"\n### {label}: {path}")
        model = PPO.load(path, device="cpu")
        rows = []
        for ep in range(args.episodes):
            r = run_episode(env, model, seed=args.seed + ep, action_ema=args.action_ema)
            rows.append(r)
            print(f"  ep{ep:03d}: success={r['success']} {r['pressed']}/5 "
                  f"steps={r['steps']} collision={r['coll_steps']} "
                  f"misuse={r['misuse_steps']} "
                  f"rate={r['action_rate']:.3f}", flush=True)
        all_rows[label] = rows

    succ_by_model = {}
    for label, rows in all_rows.items():
        succ_by_model[label] = summarize(label, rows)

    # Significance test for exactly two models
    if len(specs) == 2:
        try:
            from scipy.stats import fisher_exact
            (l1, s1), (l2, s2) = succ_by_model.items()
            table = [[int(s1.sum()), int((~s1).sum())],
                     [int(s2.sum()), int((~s2).sum())]]
            _, pval = fisher_exact(table)
            print(f"\nFisher exact ({l1} vs {l2}), successes: p = {pval:.2e}")
        except ImportError:
            pass

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            # kroki_spot LAST so index-based readers keep working
            w.writerow(["model", "epizod", "seed", "sukces", "wcisniete",
                        "kroki", "kroki_kolizji", "kroki_misuse",
                        "action_rate", "stuck", "kroki_spot"])
            for label, rows in all_rows.items():
                for ep, r in enumerate(rows):
                    w.writerow([label, ep, args.seed + ep, int(r["success"]),
                                r["pressed"], r["steps"], r["coll_steps"],
                                r["misuse_steps"],
                                f"{r['action_rate']:.5f}", int(r["stuck"]),
                                r.get("spot_steps", 0)])
        print(f"CSV saved: {args.csv}")

    env.close()


if __name__ == "__main__":
    main()

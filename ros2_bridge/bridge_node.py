#!/usr/bin/env python3
"""
Policy bridge: Webots (sim2sim) and Kinova Gen3 (hardware). Docs 6.57, 6.77.

One loop, several robot backends (robots/): dry (default), servo, ik, kortex, labpy.
Scenes: arm_d (20 Hz, 2F-85, Webots/ros2_kortex) and p8f/ps (8.065 Hz, 2F-140, lab driver).
Startup check: core FK vs robot TF (> 5 mm or > 2 deg = refuse); labpy has no TF base frame and
checks the start pose instead.

  source /opt/ros/humble/setup.bash
  ~/miniconda3/envs/robosuite/bin/python ros2_bridge/bridge_node.py --backend dry --steps 40
  # lab, policy computed but nothing sent (arm at rl_start):
  ... bridge_node.py --scene ps --backend labpy --no-send --base-height 0.20 --steps 40

--board: nominal | aruco | webots | "x,y,z[,rx,ry,rz]" in the arm base frame.
"""

import sys
sys.modules["tensorflow"] = None   # before SB3, otherwise segfault (docs 7.6)

import argparse
import csv
import os
import time

import numpy as np
import rclpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.obs_core import ObsCore, RefPlant, JointUnwrapper, SCENES, _quat2mat, wrap_pi
from core.twist_out import ActionToTwist, StaticTwist, ALPHA_ROBOSUITE, BETA_ROBOSUITE
from core.safety import wall_for
from core.webots_truth import WebotsBoardTruth
import core.step_test as ST
import core.aruco_board as AB
import core.operator_input as OP
from robots import BACKENDS, make_robot
from robots.base import GRIP_OFFSET
from board import parse_board, measure_aruco

WORLD_FILE = os.path.expanduser("~/Batchelor/webots_ros2_spot/worlds/spot.wbt")
# Flange (end_effector_link) rotation vs robosuite eef_quat, measured in Webots (6.26);
# verified by the startup check on hardware.
QUAT_SITE_REL = np.array([0.0, 0.0, -0.70710678, 0.70710678])   # xyzw


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def startup_check(node, core, unw, board_p, board_R):
    """Training FK vs robot TF. Returns (ok, report)."""
    q = unw.reset(node.q)
    _, obs = core.build(q, np.zeros(6), board_p, board_R, 0)
    eef_core = np.asarray(obs["robot0_eef_pos"]) - core.base_in_world
    quat_core = np.asarray(obs["robot0_eef_quat"])
    fp, fq = node.flange_pose()
    z = _quat2mat(fq)[:, 2]
    eef_tf = fp + GRIP_OFFSET * z
    quat_tf = quat_mul(fq, QUAT_SITE_REL)
    d_pos = float(np.linalg.norm(eef_core - eef_tf))
    cosang = abs(float(np.dot(quat_core / np.linalg.norm(quat_core), quat_tf / np.linalg.norm(quat_tf))))
    d_ang = float(np.degrees(2 * np.arccos(np.clip(cosang, -1, 1))))
    rep = (f"q from /joint_states {np.round(node.q, 4)} -> unwrapped {np.round(q, 4)}\n"
           f"  grip point: core {np.round(eef_core, 4)}  TF {np.round(eef_tf, 4)}  |d| {1000 * d_pos:.1f} mm\n"
           f"  orientation:  mismatch {d_ang:.2f} deg")
    return d_pos < 0.005 and d_ang < 2.0, rep


class RefTwist:
    """Action -> twist through the training plant (RefPlant) from the measured state."""
    needs_state = True

    def __init__(self, plant, speed_scale=1.0):
        self.plant, self.speed_scale = plant, float(speed_scale)

    def reset(self):
        self.plant.reset()

    def step(self, action, q, qd):
        v, w = self.plant.twist(q, qd, action)
        return v * self.speed_scale, w * self.speed_scale


def start_pose_check(node, core, tol=0.03):
    """Backends without a TF base frame: joints at the scene start pose (FK verified offline,
    4-9 mm vs Kortex on 18.11)."""
    err = np.abs(wrap_pi(np.asarray(node.q) - core.init_qpos))
    p, _ = core.grip_pose(node.q)
    rep = (f"q from /joint_states {np.round(node.q, 4)}; |q - start| max {np.degrees(err.max()):.2f} deg\n"
           f"  grip point (FK, base frame) {np.round(p, 4)}")
    return err.max() < tol, rep


def retreat(node, core, board_R, unw, period, dist, speed=0.1):
    """Back away from the board along its normal (base frame) by `dist` [m]."""
    v = -np.asarray(board_R, dtype=np.float64)[:, 0] * speed
    t_end = time.time() + dist / speed
    print(f"[retreat] {100 * dist:.0f} cm away from the board")
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.0)
        node.output(v, np.zeros(3), unw(node.q))
        node.spin_for(period)
    node.send_zero()


def run_step_test(node, core, args, wall, now, period, unw):
    """Twist pulses (no policy, no filter), log /joint_states, analyse.
    Start from the home pose; +A/-A pulses return near it."""
    plan = ST.pulse_plan(args.step_amp_lin, args.step_amp_ang, args.step_on)
    u, segs, T = ST.plan_to_cmd(plan)
    print(f"[step test] {len(segs)} segments, {T:.1f} s; amplitudes {args.step_amp_lin} m/s, "
          f"{args.step_amp_ang} rad/s, pulse {args.step_on} s")
    node.js_log = []
    t0 = now()
    t_next_sim = (node.sim_t or 0.0) + period
    t_next_wall = time.time() + period
    stopped = ""
    while now() - t0 < T:
        rclpy.spin_once(node, timeout_sec=0.0)
        cmd = u(now() - t0)
        v, w = cmd[:3], cmd[3:]
        if wall is not None:
            grip, _ = core.grip_pose(node.q)
            v, stop_w, d_w = wall.filter(grip, v)
            if stop_w:
                stopped = d_w
                break
        node.output(v, w, unw(node.q))
        node.wait_until(t_next_sim, t_next_wall)
        t_next_sim += period
        t_next_wall += period
    for _ in range(5):
        node.send_zero(); time.sleep(0.01)
    node.spin_for(0.5)
    samples = node.js_log
    node.js_log = None
    if stopped:
        node.get_logger().error(f"[step test] stopped by the wall: {stopped}")
    t = np.array([s_[0] for s_ in samples]) - t0
    keep = np.r_[True, np.diff(t) > 1e-6]                 # drop repeated stamps
    t = t[keep]
    poses = [core.grip_pose(s_[1]) for s_, k_ in zip(samples, keep) if k_]
    P = np.array([p for p, _ in poses]); Rm = np.array([R for _, R in poses])
    Y = ST.velocity_from_poses(t, P, Rm)
    meas = ST.analyze(t, Y, segs)
    alpha, beta, notes = ST.fit_filter(meas, ALPHA_ROBOSUITE, BETA_ROBOSUITE, period)
    raw = f"{args.step_out}_raw.csv"
    with open(raw, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t"] + [f"cmd_{a}" for a in ST.AXES] + [f"zm_{a}" for a in ST.AXES])
        for ti, yi in zip(t, Y):
            wr.writerow([f"{ti:.4f}", *np.round(u(ti), 4), *np.round(yi, 5)])
    ST.save_filter(f"{args.step_out}_filter.json", alpha, beta, meas, notes)
    print(f"\n{'axis':>4} {'dead time':>12} {'tau':>8} {'gain':>8} {'pulses':>9}")
    for ax in range(6):
        m_ = meas.get(ax)
        if m_:
            print(f"{ST.AXES[ax]:>4} {1000 * m_['L']:9.0f} ms {1000 * m_['tau']:5.0f} ms {m_['K']:8.2f} {m_['n']:9d}")
    L_med = np.nanmedian([m_["L"] for m_ in meas.values()]) if meas else np.nan
    print(f"\nmedian dead time {1000 * L_med:.0f} ms = {L_med / period:.1f} steps "
          f"(policy budget: 1 step almost free, 2 steps costly, 3 steps bad)")
    for n_ in notes:
        print("  " + n_)
    print(f"\nsaved {raw} and {args.step_out}_filter.json - use: --filter-json {args.step_out}_filter.json")
    return 0


def run_policy(node, core, model, to_twist, args, wall, board_p, board_R, board_now, unw, period):
    log = None
    if args.log_csv:
        log = csv.writer(open(args.log_csv, "w", newline=""))
        log.writerow(["t", "krok", "cel"] + [f"q{i}" for i in range(6)] + [f"a{i}" for i in range(6)]
                     + ["vx", "vy", "vz", "wx", "wy", "wz", "dystans", "align_st", "wiek_js_ms", "ik",
                        "operator"])
    # Target sequence is the supervisor's decision. Default: reached = distance <
    # reach_tol for reach_hold steps (Webots has no button physics). With a real
    # board distance misses presses (6.61): use --operator (6.63).
    seq = [int(x) for x in args.sequence.split(",")] if args.sequence else [args.target]
    # press events from a detector (std_msgs/Int32 = button index); switches on the current target
    press_evt = []
    if args.press_topic:
        from std_msgs.msg import Int32
        node.create_subscription(Int32, args.press_topic, lambda m: press_evt.append(int(m.data)), 10)
        print(f"[bridge] targets switch on {args.press_topic} (distance only warns)")
    si, hold, k_tgt, reached = 0, 0, 0, []
    op = OP.OperatorInput() if args.operator else None
    warned_near = warned_long = False
    print(f"[bridge] target sequence: {seq}")
    if op:
        print("[operator] YOU switch targets: Enter = pressed, n = skipped, q = stop "
              f"(now btn{seq[0]})", flush=True)
    print(f"{'step':>4} {'tgt':>4} {'dist':>8} {'align':>7} {'|a|':>6} {'sat':>6} {'|v|':>6} {'|w|':>6} {'js age':>8}  output")
    t_start = time.time()
    node.spin_for(0.2)
    clock_src = "sim (/clock)" if node.sim_t is not None else "wall"
    print(f"[bridge] loop at {args.rate:.0f} Hz on the {clock_src} clock")
    t_sim_next = (node.sim_t or 0.0) + period
    t_wall_next = time.time() + period
    for k in range(args.steps):
        t0 = time.time()
        rclpy.spin_once(node, timeout_sec=0.0)
        age = 1000 * (time.time() - node.js_stamp)   # after spin_once, else negative
        q = unw(node.q)
        bp, bR = board_now()
        if bp is not None:
            board_p, board_R = bp, bR
        tgt = seq[si]
        flat, obs = core.build(q, node.qd, board_p, board_R, tgt)
        a, _ = model.predict(flat, deterministic=True)
        v, w = (to_twist.step(a, q, node.qd) if getattr(to_twist, "needs_state", False)
                else to_twist.step(a))
        out = ""
        if wall is not None:
            grip = np.asarray(obs["robot0_eef_pos"]) - core.base_in_world
            v, stop_w, d_w = wall.filter(grip, v)
            if stop_w:
                node.send_zero()
                node.get_logger().error(f"[wall] {d_w} - end of run")
                break
            out = d_w
        out = (out + " " + node.output(v, w, q)).strip()
        dist = float(obs["target_distance"][0])
        align = float(np.degrees(obs["eef_align_with_target"][0]))
        ev = op.poll() if op else None
        if k % 5 == 0 or k == args.steps - 1:
            print(f"{k:4d} {tgt:4d} {dist:8.3f} {align:6.1f}° {np.abs(a[:6]).mean():6.2f} "
                  f"{100 * np.mean(np.abs(a[:6]) > 0.99):5.0f}% {np.linalg.norm(v):6.3f} "
                  f"{np.linalg.norm(w):6.3f} {age:7.0f}ms  {out}")
        if log:
            log.writerow([f"{t0 - t_start:.4f}", k, tgt, *np.round(q, 5), *np.round(a[:6], 4),
                          *np.round(v, 4), *np.round(w, 4), f"{dist:.4f}", f"{align:.2f}", f"{age:.1f}", out,
                          ev[0] if ev else ""])
        # target switching
        k_tgt += 1
        hold = hold + 1 if dist < args.reach_tol else 0
        if op:
            # distance and step limit only hint; the operator decides
            if hold >= args.reach_hold and not warned_near:
                print(f"[operator] btn{tgt}: grip point < {1000 * args.reach_tol:.0f} mm "
                      f"(step {k_tgt}) - has the cap latched?", flush=True)
                warned_near = True
            if k_tgt >= args.max_steps_per_target and not warned_long:
                print(f"[operator] btn{tgt}: already {k_tgt} steps - n = skipped, q = stop", flush=True)
                warned_long = True
            if ev is None:
                node.wait_until(t_sim_next, t_wall_next)
                t_sim_next += period
                t_wall_next += period
                continue
            if ev[0] == OP.STOP:
                print(f"[operator] STOP at btn{tgt} after {k_tgt} steps", flush=True)
                break
            ok_t = ev[0] == OP.PRESSED
            reached.append((tgt, ok_t, k_tgt))
            print(f"[operator] btn{tgt}: {ev[0].upper()} after {k_tgt} steps ({k_tgt / args.rate:.1f} s), "
                  f"distance {1000 * dist:.0f} mm", flush=True)
            si, hold, k_tgt, warned_near, warned_long = si + 1, 0, 0, False, False
            if si >= len(seq):
                break
            print(f"[operator] now btn{seq[si]}", flush=True)
        elif press_evt and tgt in press_evt or (not args.press_topic and hold >= args.reach_hold) \
                or k_tgt >= args.max_steps_per_target:
            ok_t = (tgt in press_evt) if args.press_topic else hold >= args.reach_hold
            reached.append((tgt, ok_t, k_tgt))
            print(f"[supervisor] btn{tgt}: {('PRESSED' if args.press_topic else 'REACHED') if ok_t else 'STEP LIMIT'} "
                  f"after {k_tgt} steps ({k_tgt / args.rate:.1f} s), grip point {1000 * dist:.0f} mm from the cap")
            press_evt.clear()
            si, hold, k_tgt = si + 1, 0, 0
            if si >= len(seq):
                break
        node.wait_until(t_sim_next, t_wall_next)
        t_sim_next += period
        t_wall_next += period
    if hasattr(node, "n_deadband"):
        print(f"[bridge] commands in the driver dead band: {node.n_deadband}/{k + 1} steps")
    if len(seq) > 1 or reached:
        print(f"[supervisor] result: {sum(1 for _, ok_t, _ in reached if ok_t)}/{len(seq)} targets "
              f"{'pressed (operator)' if op else 'reached'}: "
              + ", ".join(f"btn{t} {'OK' if o else '--'} {n} st." for t, o, n in reached))
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=BACKENDS, default="dry")
    ap.add_argument("--scene", choices=list(SCENES), default="arm_d",
                    help="training scene contract: arm_d (Webots/ros2_kortex) | p8f | ps (lab, docs 6.77)")
    ap.add_argument("--model", default=None, help="checkpoint instead of the scene's model")
    ap.add_argument("--twist-map", choices=["ref", "static"], default="ref",
                    help="[p8f/ps] action -> twist: training plant from the measured state (ref) or "
                         "the static 6x6 gain matrix (static)")
    # ---- loop and targets
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--rate", type=float, default=None, help="loop rate [Hz]; default = training rate of the scene")
    ap.add_argument("--target", type=int, default=0, help="target button index (0..4)")
    ap.add_argument("--sequence", default=None,
                    help="target order, e.g. 2,1,4,0,3 (easiest first); overrides --target")
    ap.add_argument("--reach-tol", type=float, default=0.015,
                    help="[sequence] target reached when grip point to cap distance < tol [m]")
    ap.add_argument("--reach-hold", type=int, default=3,
                    help="[sequence] consecutive steps below tol")
    ap.add_argument("--max-steps-per-target", type=int, default=400)
    ap.add_argument("--operator", action="store_true",
                    help="[sequence] the OPERATOR switches targets (Enter = pressed, n = skipped, q = stop); "
                         "distance and step limit only warn (docs 6.63)")
    ap.add_argument("--press-topic", default=None,
                    help="[sequence] std_msgs/Int32 press events (button index) switch targets, e.g. /estop_pressed")
    ap.add_argument("--log-csv", default=None)
    # ---- board
    ap.add_argument("--base-height", type=float, default=None,
                    help="[--board nominal] arm base height above the floor [m]: board placed by the training "
                         "stand-off rule (lying Spot 0.20); the real board pose should be measured")
    ap.add_argument("--board", default="nominal",
                    help="nominal | aruco | webots (world truth + TF odom) | x,y,z[,rx,ry,rz] in the base frame")
    ap.add_argument("--world", default=WORLD_FILE, help="[--board webots] .wbt file")
    ap.add_argument("--aruco-frames", type=int, default=20, help="[--board aruco] frames to average")
    ap.add_argument("--cam-extrinsic", default="webots", choices=["webots"],
                    help="[--board aruco] camera extrinsic in the flange frame; hardware: ADD "
                         "from the Kinova Vision module calibration")
    ap.add_argument("--compare-truth", action="store_true",
                    help="[--board aruco, Webots] compare pose, observation and action with the truth")
    # ---- startup
    ap.add_argument("--home", action="store_true", help="move to arm_d init_qpos before the loop")
    ap.add_argument("--home-time", type=float, default=8.0)
    ap.add_argument("--home-safe", action="store_true",
                    help="homing along a collision-checked path: straighten -> wrist -> init_qpos (implies --home)")
    ap.add_argument("--skip-check", action="store_true", help="skip the FK vs TF check (NOT on hardware)")
    # ---- output and safety
    ap.add_argument("--speed-scale", type=float, default=1.0,
                    help="<1 = reduced caps for first runs (changes dynamics!)")
    ap.add_argument("--filter-json", default=None,
                    help="action->twist filter from the step test (instead of robosuite coefficients)")
    ap.add_argument("--watchdog", type=float, default=0.1, help="[s] without a command -> zero twist")
    ap.add_argument("--wall-x", type=float, default=-0.015,
                    help="virtual wall: grip-point x in the board frame [m] (0 = back plate plane)")
    ap.add_argument("--wall-brake", type=float, default=0.25,
                    help="virtual wall: braking time [s]; after the step test ~2.5*(L+tau)")
    ap.add_argument("--no-wall", action="store_true", help="disable the virtual wall (NOT in the in-air test)")
    # ---- kortex
    ap.add_argument("--tool-tcp-z", type=float, default=GRIP_OFFSET,
                    help="[kortex] Kinova Web App tool TCP offset along flange Z [m]")
    ap.add_argument("--no-send", action="store_true",
                    help="[labpy] compute obs/actions, publish nothing (no motion, no /goto_position)")
    ap.add_argument("--retreat", type=float, default=None,
                    help="[labpy] after the loop back away from the board by this many metres (default 0.08)")
    ap.add_argument("--no-switch", action="store_true",
                    help="[kortex] do not switch controllers (JTC for homing, twist for driving)")
    # ---- step test
    ap.add_argument("--step-test", action="store_true",
                    help="instead of the policy: twist pulses on each axis, response measurement, new filter")
    ap.add_argument("--step-amp-lin", type=float, default=0.05, help="[step-test] linear amplitude [m/s]")
    ap.add_argument("--step-amp-ang", type=float, default=0.2, help="[step-test] angular amplitude [rad/s]")
    ap.add_argument("--step-on", type=float, default=0.8, help="[step-test] pulse duration [s]")
    ap.add_argument("--step-out", default="step_test",
                    help="[step-test] output prefix: <prefix>_raw.csv, <prefix>_filter.json")
    # ---- ik (Webots)
    ap.add_argument("--ik-vel", action="store_true", help="[ik] send velocities in trajectory points")
    ap.add_argument("--ik-ref", action="store_true", default=True,
                    help="[ik] integrate the reference pose in the bridge (default ON)")
    ap.add_argument("--no-ik-ref", dest="ik_ref", action="store_false")
    ap.add_argument("--ik-ref-max", type=float, default=0.15,
                    help="[ik] max |q_ref - q_measured| [rad]")
    ap.add_argument("--ik-stream", type=int, default=3,
                    help="[ik] trajectory points per command (streaming); 0 = single point")
    ap.add_argument("--ik-lead", type=float, default=1.0,
                    help="[ik] per-step shift multiplier (1 = v*dt; >1 compensates controller lag)")
    ap.add_argument("--ik-horizon", type=float, default=0.05,
                    help="[ik] trajectory point time_from_start [s]; = loop period")
    args = ap.parse_args()
    args.home = args.home or args.home_safe
    return args


def main():
    args = parse_args()

    from stable_baselines3 import PPO
    scene = SCENES[args.scene]
    model_path = args.model or scene["model"]
    print(f"[bridge] loading the core (MuJoCo, {args.scene} scene) and the policy {model_path}...")
    core = ObsCore(scene)
    model = PPO.load(model_path, device="cpu")
    args.rate = args.rate or core.rate
    if abs(args.rate - core.rate) > 1e-6:
        print(f"[bridge] WARNING: loop {args.rate:.3f} Hz, training {core.rate:.3f} Hz")
    if "cfg" in scene and args.twist_map == "ref":
        to_twist = RefTwist(RefPlant(scene), speed_scale=args.speed_scale)
        print(f"[bridge] action -> twist: training plant from the measured state (EMA, OSC, caps "
              f"{core.twist_caps}); grip point {core.grip_offset:.4f} m from the flange")
    elif "twist_gain" in scene:
        to_twist = StaticTwist(scene["twist_gain"], ema=core.action_ema, v_cap=core.twist_caps[0],
                               w_cap=core.twist_caps[1], speed_scale=args.speed_scale)
        print(f"[bridge] action -> twist: EMA {core.action_ema}, gain {np.asarray(scene['twist_gain']).shape}, "
              f"caps {core.twist_caps}, grip point {core.grip_offset:.4f} m from the flange")
    elif args.filter_json:
        fa, fb = ST.load_filter(args.filter_json)
        to_twist = ActionToTwist(alpha=fa, beta=fb, speed_scale=args.speed_scale)
        print(f"[bridge] filter from {args.filter_json}: alpha {np.round(fa, 3)}, beta {np.round(fb, 4)}")
    else:
        to_twist = ActionToTwist(speed_scale=args.speed_scale)
    board_p, board_R = parse_board(args.board, core)
    if args.base_height is not None and args.board == "nominal":
        board_p, board_R = core.posture_board_pose(args.base_height)
        print(f"[bridge] base {args.base_height:.2f} m above the floor -> board by the stand-off rule")
    truth = truth_cmp = None
    if args.board == "aruco":
        if args.compare_truth:
            truth_cmp = WebotsBoardTruth(args.world, core.btn_offsets, core.btn_normals)
        T_fc = {"webots": (AB.P_FLANGE_CAM_WEBOTS, AB.R_FLANGE_CAM_WEBOTS)}[args.cam_extrinsic]
        print(f"[aruco] marker {AB.DICT} id {AB.MARKER_ID}, {1000 * AB.MARKER_SIZE:.0f} mm; "
              f"camera extrinsic: {args.cam_extrinsic}")
    elif args.board == "webots":
        truth = WebotsBoardTruth(args.world, core.btn_offsets, core.btn_normals)
        print("[webots] " + truth.report())
        nom_p, _ = core.nominal_board_pose()
        bl, yaw = truth.stand_pose_world(nom_p)
        print(f"[webots] training stand-off: base_link world {np.round(bl[:2], 3)}, "
              f"yaw {np.degrees(yaw):.1f} deg  (in odom: {np.round([-(bl[0] - truth.spot_p0[0]), -(bl[1] - truth.spot_p0[1])], 3)}, yaw 0)")
    else:
        btn, _ = core.buttons_base(board_p, board_R)
        print(f"[bridge] board in the base frame: centre {np.round(board_p, 4)}; target btn{args.target} "
              f"{np.round(btn[args.target], 4)}")

    rclpy.init()
    node = make_robot(args, core)
    try:
        # ---- 1. robot state
        if not node.wait_ready():
            node.get_logger().error("no /joint_states" + ("" if getattr(node, "fk_only", False) else
                                    f" or TF {node.prof['base']}->{node.prof['flange']}"))
            return 1
        # ---- 2. homing
        if not node.prepare_homing():
            return 1
        if args.home:
            if args.home_safe:
                node.spin_for(0.5)
                wps, rap = core.safe_home_path(node.q)
                print(f"[homing] from {np.round(node.q, 3)}: {rap}")
                if wps is None:
                    node.get_logger().error("homing aborted - sending nothing"); return 1
                for wp in wps[:-1]:
                    node.home(wp, args.home_time, quiet=True)
            if not node.home(core.init_qpos, args.home_time):
                return 1

        # ---- 3. board pose
        def board_now():
            if truth is None:
                return board_p, board_R
            T = node.base_from("odom")
            if T is None:
                return None, None
            return truth.in_base(*T)

        if args.board == "aruco":
            if args.backend != "dry" and not args.home:
                node.get_logger().warn("ArUco measurement without --home: the arm must be STILL")
            bp, bR, rap = measure_aruco(node, args.aruco_frames, T_fc)
            if bp is None:
                node.get_logger().error(f"[aruco] {rap}"); return 1
            nom_p, _ = core.nominal_board_pose()
            print(f"[aruco] {rap}\n[aruco] board in the base frame: centre {np.round(bp, 4)} "
                  f"(training {np.round(nom_p, 4)}; difference {np.round(1000 * (bp - nom_p), 0)} mm), "
                  f"normal {np.round(bR[:, 0], 3)}")
            board_p, board_R = bp, bR
            node.drop_camera()
            if truth_cmp is not None:
                T = node.base_from("odom")
                pt, Rt = truth_cmp.in_base(*T)
                ba, _ = core.buttons_base(bp, bR)
                bt, _ = core.buttons_base(pt, Rt)
                print(f"[aruco vs truth] centre {1000 * np.linalg.norm(bp - pt):.1f} mm, orientation "
                      f"{AB.rot_err_deg(bR, Rt):.2f} deg, caps [mm] "
                      f"{np.round(1000 * np.linalg.norm(ba - bt, axis=1), 1).tolist()}")
                q0 = JointUnwrapper(core.init_qpos).reset(node.q)
                for tgt in range(5):
                    fa, _ = core.build(q0, np.zeros(6), bp, bR, tgt)
                    ft, _ = core.build(q0, np.zeros(6), pt, Rt, tgt)
                    aa, _ = model.predict(fa, deterministic=True)
                    at, _ = model.predict(ft, deterministic=True)
                    print(f"  btn{tgt}: |obs| max {np.max(np.abs(fa - ft)):.4f} "
                          f"(target_rel {1000 * np.max(np.abs(fa[47:50] - ft[47:50])):.1f} mm, "
                          f"align {np.degrees(abs(fa[59] - ft[59])):.2f} deg), |action| max {np.max(np.abs(aa - at)):.3f}")
        if truth is not None:
            bp, bR = board_now()
            if bp is None:
                node.get_logger().error("no TF base <- odom"); return 1
            nom_p, _ = core.nominal_board_pose()
            dyaw = np.degrees(np.arctan2(bR[1, 0], bR[0, 0]))
            print(f"[webots] board in the base frame: centre {np.round(bp, 4)} "
                  f"(training {np.round(nom_p, 4)}; difference {np.round(1000 * (bp - nom_p), 0)} mm), "
                  f"rotation about Z {dyaw:+.1f} deg (training 0)")
            board_p, board_R = bp, bR

        # ---- 4. FK check
        unw = JointUnwrapper(core.init_qpos)
        if getattr(node, "fk_only", False):
            ok, rep = start_pose_check(node, core)
            print("[start pose check]\n  " + rep)
            if not ok and not args.skip_check:
                node.get_logger().error("arm is not at the scene start pose (rl_start) - use --home")
                return 1
        else:
            ok, rep = startup_check(node, core, unw, board_p, board_R)
            print("[FK check]\n  " + rep)
            if not ok and not args.skip_check:
                node.get_logger().error("core and robot DISAGREE on kinematics - stop")
                return 1

        # ---- 5. output on
        if not node.start_output():
            return 1
        if node.streams_twist:
            node.spin_for(1.0)
            if args.home:
                drift = np.abs(np.angle(np.exp(1j * (node.q - core.init_qpos))))
                print(f"[bridge] drift after output start [rad]: {np.round(drift, 3)}")
                if drift.max() > 0.03:
                    node.get_logger().error("the arm left the start pose before the loop - stop")
                    return 1
            n_sub = node.tw_pub.get_subscription_count()
            print(f"[bridge] {node.prof['twist']}: subscribers {n_sub}" + node.output_status())
            if n_sub == 0:
                node.get_logger().error("nobody listens to the twist - Servo/twist_controller not active")
                return 1
            # must exceed the loop period, or zeros go out between steps (8 Hz: 124 ms)
            node.watchdog(max(args.watchdog, 1.5 / args.rate))
        wall = None
        if not args.no_wall:
            wall = wall_for(core, board_p, board_R, x_wall=args.wall_x, t_brake=args.wall_brake)
            print(f"[wall] grip point no deeper than {1000 * -args.wall_x:.0f} mm before the board back "
                  f"plate; box +-{1000 * wall.half:.0f} mm; no limit on the Spot side (the body is physical)")
        period = 1.0 / args.rate

        def now():
            return node.sim_t if node.sim_t is not None else time.time()

        # ---- 6. loop
        if args.step_test:
            return run_step_test(node, core, args, wall, now, period, unw)
        rc = run_policy(node, core, model, to_twist, args, wall, board_p, board_R, board_now, unw, period)
        dist = args.retreat if args.retreat is not None else (0.08 if args.backend == "labpy" else 0.0)
        if dist > 0 and getattr(node, "send", False):
            retreat(node, core, board_R, unw, period, dist)
        return rc
    except KeyboardInterrupt:
        print("\n[bridge] interrupted")
    finally:
        node.stop_all()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

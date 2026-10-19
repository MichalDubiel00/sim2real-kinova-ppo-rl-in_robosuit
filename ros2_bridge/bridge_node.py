#!/usr/bin/env python3
"""
arm_d policy bridge: Webots (sim2sim) and Kinova Gen3 (hardware). Docs 6.57.

One core (ObsCore + JointUnwrapper + ActionToTwist), several outputs:
  --backend dry     compute obs and actions, send nothing (default)
  --backend servo   Webots: MoveIt Servo, TwistStamped in the base frame
  --backend ik      Webots: Jacobian steps streamed to the JTC
  --backend kortex  hardware: ros2_kortex twist_controller, Twist in the TOOL frame

Startup check: core FK vs robot TF; > 5 mm or > 2 deg and the bridge refuses to run.

  source /opt/ros/humble/setup.bash
  ~/miniconda3/envs/robosuite/bin/python ros2_bridge/bridge_node.py --backend dry --steps 40

--board: nominal | aruco | webots | "x,y,z[,rx,ry,rz]" in the arm base frame.
"""

import sys
sys.modules["tensorflow"] = None   # before SB3, otherwise segfault (docs 7.6)

import argparse
import csv
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_msgs.msg import Int8, Empty
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Trigger
from geometry_msgs.msg import Twist, TwistStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obs_core import ObsCore, JointUnwrapper, SCENE_ARM_D, _quat2mat
from twist_out import ActionToTwist, base_to_tool, shift_point, ALPHA_ROBOSUITE, BETA_ROBOSUITE
from safety import wall_for
import step_test as ST
from webots_truth import WebotsBoardTruth
import aruco_board as AB
import operator_input as OP

WORLD_FILE = os.path.expanduser("~/Batchelor/webots_ros2_spot/worlds/spot.wbt")

JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
GRIP_OFFSET = 0.1448      # robosuite grip point along flange Z (6.26)
# Flange (end_effector_link) rotation vs robosuite eef_quat, measured in Webots (6.26);
# verified by the startup check on hardware.
QUAT_SITE_REL = np.array([0.0, 0.0, -0.70710678, 0.70710678])   # xyzw

PROFILES = {
    "servo": dict(base="gen3_base_link", flange="end_effector_link",
                  cm="/controller_manager", jtc_name="kinova_joint_trajectory_controller",
                  twist_name=None,          # Webots has no twist_controller
                  jtc="/kinova_joint_trajectory_controller/joint_trajectory",
                  twist="/servo_node/delta_twist_cmds",
                  img="/Gen3/kinova_color/image_color", info="/Gen3/kinova_color/camera_info"),
    # kinova_vision defaults - VERIFY on the robot (ros2 topic list)
    "kortex": dict(base="base_link", flange="end_effector_link",
                   cm="/controller_manager", jtc_name="joint_trajectory_controller",
                   twist_name="twist_controller",
                   jtc="/joint_trajectory_controller/joint_trajectory",
                   twist="/twist_controller/commands",
                   img="/camera/color/image_raw", info="/camera/color/camera_info"),
}
PROFILES["dry"] = PROFILES["servo"]
PROFILES["ik"] = PROFILES["servo"]   # Webots: /compute_ik + JointTrajectory

# moveit_servo (Humble) StatusCode
SERVO_STATUS = {0: "OK", 1: "slowing: singularity", 2: "STOP: singularity",
                3: "slowing: collision", 4: "STOP: collision", 5: "STOP: joint limit"}


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def axang_to_R(v):
    v = np.asarray(v, dtype=np.float64)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class Bridge(Node):
    def __init__(self, args):
        super().__init__("arm_d_bridge")
        self.args = args
        self.prof = PROFILES[args.backend]
        self.q = self.qd = None
        self.js_stamp = None
        self.create_subscription(JointState, "/joint_states", self._js, 10)
        self.tfb = Buffer()
        self.tfl = TransformListener(self.tfb, self)
        self.jtc_pub = self.create_publisher(JointTrajectory, self.prof["jtc"], 10)
        self.servo_status = None
        if args.backend == "servo":
            self.tw_pub = self.create_publisher(TwistStamped, self.prof["twist"], 10)
            self.create_subscription(Int8, "/servo_node/status",
                                     lambda m: setattr(self, "servo_status", int(m.data)), 10)
        elif args.backend == "kortex":
            self.tw_pub = self.create_publisher(Twist, self.prof["twist"], 10)
        elif args.backend == "ik":
            self.tw_pub = None
        else:
            self.tw_pub = None
        # heartbeat for deadman.py: main loop only, never the watchdog thread
        self.hb_pub = self.create_publisher(Empty, "/arm_d_bridge/heartbeat", 10)
        self.img = self.cam_info = None
        self._cam_subs = []
        if args.board == "aruco":
            self._cam_subs = [
                self.create_subscription(Image, self.prof["img"], lambda m: setattr(self, "img", m), 1),
                self.create_subscription(CameraInfo, self.prof["info"], lambda m: setattr(self, "cam_info", m), 1)]
        # Webots: pace on sim time (at RTF 0.70 wall pacing would run the policy at
        # ~29 Hz sim time). Hardware has no /clock -> wall clock.
        self.sim_t = None
        self.create_subscription(Clock, "/clock",
                                 lambda m: setattr(self, "sim_t", m.clock.sec + m.clock.nanosec * 1e-9), 10)
        self._last_cmd_t = time.time()
        self._lock = threading.Lock()
        self._stop = False
        self.js_log = None

    # --------------------------------------------------------- input --
    def _js(self, msg):
        if not all(j in msg.name for j in JOINTS):
            return
        idx = [msg.name.index(j) for j in JOINTS]
        self.q = np.array([msg.position[i] for i in idx])
        self.qd = (np.array([msg.velocity[i] for i in idx]) if len(msg.velocity) >= len(msg.name)
                   else np.zeros(6))
        self.js_stamp = time.time()
        if self.js_log is not None:          # step test: every sample with its stamp
            self.js_log.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, self.q.copy()))

    def drop_camera(self):
        """Unsubscribe the camera after measuring so webots_ros2 stops rendering it."""
        for sub in self._cam_subs:
            self.destroy_subscription(sub)
        self._cam_subs = []

    def wait_until(self, t_sim_target, t_wall_target):
        """Wait until a time on the sim clock if present, else the wall clock."""
        while True:
            rclpy.spin_once(self, timeout_sec=0.001)
            if self.sim_t is not None:
                if self.sim_t >= t_sim_target:
                    return
            elif time.time() >= t_wall_target:
                return

    def spin_for(self, sec):
        """Not time.sleep(): that does not process messages (6.26)."""
        t0 = time.time()
        while time.time() - t0 < sec:
            rclpy.spin_once(self, timeout_sec=0.005)

    def wait_ready(self, timeout=15.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.q is not None and self.flange_pose() is not None:
                return True
        return False

    def flange_pose(self):
        try:
            tr = self.tfb.lookup_transform(self.prof["base"], self.prof["flange"], rclpy.time.Time())
        except Exception:
            return None
        t, r = tr.transform.translation, tr.transform.rotation
        return np.array([t.x, t.y, t.z]), np.array([r.x, r.y, r.z, r.w])

    def start_servo(self, timeout=5.0, srv="start_servo"):
        """Call /servo_node/start_servo (or stop_servo). Without start Servo does
        nothing. start on a running Servo pulls the arm back to its stale pose
        (15.10), so the bridge does stop -> home -> start."""
        cli = self.create_client(Trigger, f"/servo_node/{srv}")
        if not cli.wait_for_service(timeout_sec=timeout):
            return False, f"service /servo_node/{srv} unavailable"
        fut = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        r = fut.result()
        if r is None:
            return False, "no response"
        return bool(r.success), r.message

    # --------------------------------------- controllers (ros2_control) --
    def _cm_call(self, srv_type, name, req, timeout=5.0):
        cli = self.create_client(srv_type, f"{self.prof['cm']}/{name}")
        if not cli.wait_for_service(timeout_sec=timeout):
            return None
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result()

    def active_controllers(self):
        from controller_manager_msgs.srv import ListControllers
        r = self._cm_call(ListControllers, "list_controllers", ListControllers.Request())
        if r is None:
            return None
        return {c.name: c.state for c in r.controller}

    def use_controller(self, which):
        """which = 'jtc' (homing, holding) or 'twist' (policy). Both claim the same
        joint interfaces, so only one is active. Zero twist is sent before handing
        over to the JTC (twist_controller holds its last command). Returns (ok, msg)."""
        jtc, tw = self.prof["jtc_name"], self.prof["twist_name"]
        if tw is None:
            return True, "profile without twist_controller - no switching"
        from controller_manager_msgs.srv import SwitchController
        state = self.active_controllers()
        if state is None:
            return False, f"no {self.prof['cm']}/list_controllers"
        want, other = (jtc, tw) if which == "jtc" else (tw, jtc)
        if state.get(want) == "active" and state.get(other) != "active":
            return True, f"{want} already active"
        if which == "jtc":
            for _ in range(5):
                self.send_zero(); time.sleep(0.01)
        req = SwitchController.Request()
        req.activate_controllers = [want]
        req.deactivate_controllers = [other] if state.get(other) == "active" else []
        req.strictness = SwitchController.Request.STRICT
        req.activate_asap = True
        req.timeout.sec = 2
        r = self._cm_call(SwitchController, "switch_controller", req)
        if r is None or not r.ok:
            return False, f"switch_controller refused ({want} <- {other}); state: {state}"
        st = self.active_controllers() or {}
        return st.get(want) == "active", f"{want} active, {other}: {st.get(other)}"

    def base_from(self, frame):
        """(p, R) of base <- frame from TF, or None."""
        try:
            tr = self.tfb.lookup_transform(self.prof["base"], frame, rclpy.time.Time())
        except Exception:
            return None
        t, r = tr.transform.translation, tr.transform.rotation
        return np.array([t.x, t.y, t.z]), _quat2mat(np.array([r.x, r.y, r.z, r.w]))

    # -------------------------------------------------------- output --
    def send_twist(self, v_base, w_base):
        """Send a twist; v_base/w_base refer to the GRIP POINT in the base frame."""
        if self.tw_pub is None:
            return
        with self._lock:
            if self.args.backend == "servo":
                m = TwistStamped()
                m.header.stamp = self.get_clock().now().to_msg()
                m.header.frame_id = self.prof["base"]
                # Servo is unitless: cmd = v / scale (5.0 m/s, 2.0 rad/s); it controls
                # the flange, so shift the velocity from the grip point
                fp = self.flange_pose()
                if fp is not None:
                    z = _quat2mat(fp[1])[:, 2]
                    v_base, w_base = shift_point(v_base, w_base, -GRIP_OFFSET * z)
                (m.twist.linear.x, m.twist.linear.y, m.twist.linear.z) = map(float, np.asarray(v_base) / 5.0)
                (m.twist.angular.x, m.twist.angular.y, m.twist.angular.z) = map(float, np.asarray(w_base) / 2.0)
            else:
                fp = self.flange_pose()
                if fp is None:
                    self._send_zero_locked()
                    return
                R = _quat2mat(fp[1])           # flange orientation = firmware TOOL frame
                z = R[:, 2]                    # (assumes an unrotated Web App tool - verify)
                # firmware controls the Web App TCP, the policy the grip point
                v_base, w_base = shift_point(v_base, w_base,
                                             (self.args.tool_tcp_z - GRIP_OFFSET) * z)
                v_t, w_t = base_to_tool(v_base, w_base, R)
                m = Twist()
                (m.linear.x, m.linear.y, m.linear.z) = map(float, v_t)
                (m.angular.x, m.angular.y, m.angular.z) = map(float, w_t)
            self.tw_pub.publish(m)
            self._last_cmd_t = time.time()
            self.hb_pub.publish(Empty())

    # ------------------------------------------------------ IK path --
    def send_joint_stream(self, q_now, dq, period, n_points):
        """Stream to the JTC: points q + k*dq at k*period, k = 1..n. Each command
        replaces the trajectory before it ends, so the arm never brakes per step
        (a single point realised ~14 % of the step)."""
        msg = JointTrajectory()
        msg.joint_names = JOINTS
        for k in range(1, n_points + 1):
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in np.asarray(q_now) + k * np.asarray(dq)]
            t = k * period
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t % 1) * 1e9)
            msg.points.append(pt)
        self.jtc_pub.publish(msg)

    def send_joint_step(self, q_cmd, duration, qd_cmd=None):
        """'ik' path (Webots demo): joint targets -> JointTrajectory. qd_cmd sets
        point velocities; without them every replacement restarts from zero."""
        msg = JointTrajectory()
        msg.joint_names = JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in q_cmd]
        if qd_cmd is not None:
            pt.velocities = [float(x) for x in qd_cmd]
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = int((duration % 1) * 1e9)
        msg.points.append(pt)
        self.jtc_pub.publish(msg)

    def _send_zero_locked(self):
        if self.tw_pub is None:
            return
        if self.args.backend == "servo":
            m = TwistStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = self.prof["base"]
        else:
            m = Twist()
        self.tw_pub.publish(m)

    def send_zero(self):
        with self._lock:
            self._send_zero_locked()

    def watchdog(self, timeout):
        """Thread: send zeros if the loop has not commanded for `timeout` s
        (twist_controller holds the last twist)."""
        def run():
            while not self._stop:
                if time.time() - self._last_cmd_t > timeout:
                    self.send_zero()
                time.sleep(timeout / 4)
        th = threading.Thread(target=run, daemon=True)
        th.start()
        return th

    def home(self, qpos, duration, quiet=False):
        # Homing also runs in dry mode: the FK check needs the start pose.
        t0 = time.time()
        while self.jtc_pub.get_subscription_count() == 0 and time.time() - t0 < 5.0:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.jtc_pub.get_subscription_count() == 0:
            self.get_logger().error(f"nobody subscribes to {self.prof['jtc']} "
                                    "(on Kortex: is joint_trajectory_controller active?)")
            return False
        msg = JointTrajectory()
        msg.joint_names = JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in qpos]
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = int((duration % 1) * 1e9)
        msg.points.append(pt)
        self.jtc_pub.publish(msg)
        self.spin_for(duration + 1.0)
        err = np.abs(np.angle(np.exp(1j * (self.q - np.asarray(qpos)))))
        if err.max() > 0.02:
            if not quiet:
                self.get_logger().error(f"homing failed, error [rad] {np.round(err, 3)}")
            return False
        return True


def parse_board(s, core):
    if s == "nominal":
        return core.nominal_board_pose()
    if s in ("webots", "aruco"):
        return None, None          # webots: TF in the loop; aruco: measured after homing
    v = [float(x) for x in s.split(",")]
    if len(v) == 3:
        return np.array(v), np.eye(3)
    if len(v) == 6:
        return np.array(v[:3]), axang_to_R(v[3:])
    raise SystemExit("--board: 'nominal', 'x,y,z' or 'x,y,z,rx,ry,rz'")


def _to_bgr(m):
    import cv2
    a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
    if m.encoding in ("bgra8", "bgr8"):
        return np.ascontiguousarray(a[:, :, :3])
    return cv2.cvtColor(np.ascontiguousarray(a[:, :, :3]), cv2.COLOR_RGB2BGR)


def measure_aruco(node, frames, T_fc, timeout=30.0):
    """Board pose from ArUco: mean of N frames on a still arm (constant per episode,
    as in training). Returns (p, R, report) or (None, None, reason)."""
    t0, last, poses, errs = time.time(), None, [], []
    while len(poses) < frames and time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.05)
        m, ci = node.img, node.cam_info
        if m is None or ci is None:
            continue
        st = (m.header.stamp.sec, m.header.stamp.nanosec)
        if st == last:
            continue
        last = st
        T_bf = node.flange_pose()
        if T_bf is None:
            continue
        T_bf = (T_bf[0], _quat2mat(T_bf[1]))
        K = np.array(ci.k).reshape(3, 3)
        D = np.array(ci.d) if len(ci.d) else np.zeros(5)
        r = AB.detect(_to_bgr(m), K, D)
        if r is None:
            continue
        ch = AB.choose_candidate(r[0], AB.compose(T_bf, T_fc))
        if ch is None:
            continue
        poses.append(AB.board_in_base(T_bf, T_fc, (ch[0], ch[1])))
        errs.append(ch[2])
    if not poses:
        return None, None, "marker not detected (camera? lighting? marker in view?)"
    p, R = AB.average_poses(poses)
    P = np.array([q for q, _ in poses])
    rot = max(AB.rot_err_deg(Ri, R) for _, Ri in poses)
    return p, R, (f"{len(poses)}/{frames} frames, reprojection med. {np.median(errs):.2f} px, "
                  f"spread {1000 * P.std(0).max():.2f} mm / {rot:.2f} deg")


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


def run_step_test(node, core, args, output, wall, now, period, unw):
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
        output(v, w, unw(node.q))
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["dry", "servo", "kortex", "ik"], default="dry")
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
    ap.add_argument("--board", default="nominal",
                    help="nominal | aruco | webots (world truth + TF odom) | x,y,z[,rx,ry,rz] in the base frame")
    ap.add_argument("--world", default=WORLD_FILE, help="[--board webots] .wbt file")
    ap.add_argument("--aruco-frames", type=int, default=20, help="[--board aruco] frames to average")
    ap.add_argument("--cam-extrinsic", default="webots", choices=["webots"],
                    help="[--board aruco] camera extrinsic in the flange frame; hardware: ADD "
                         "from the Kinova Vision module calibration")
    ap.add_argument("--compare-truth", action="store_true",
                    help="[--board aruco, Webots] compare pose, observation and action with the truth")
    ap.add_argument("--target", type=int, default=0, help="target button index (0..4)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--rate", type=float, default=20.0, help="20 Hz as in training")
    ap.add_argument("--home", action="store_true", help="move to arm_d init_qpos before the loop")
    ap.add_argument("--home-time", type=float, default=8.0)
    ap.add_argument("--home-safe", action="store_true",
                    help="homing along a collision-checked path: straighten -> wrist -> init_qpos")
    ap.add_argument("--speed-scale", type=float, default=1.0,
                    help="<1 = reduced caps for first runs (changes dynamics!)")
    ap.add_argument("--tool-tcp-z", type=float, default=GRIP_OFFSET,
                    help="[kortex] Kinova Web App tool TCP offset along flange Z [m]")
    ap.add_argument("--watchdog", type=float, default=0.1, help="[s] without a command -> zero twist")
    ap.add_argument("--skip-check", action="store_true", help="skip the FK vs TF check (NOT on hardware)")
    ap.add_argument("--log-csv", default=None)
    ap.add_argument("--no-switch", action="store_true",
                    help="[kortex] do not switch controllers (JTC for homing, twist for driving)")
    ap.add_argument("--wall-x", type=float, default=-0.015,
                    help="virtual wall: grip-point x in the board frame [m] (0 = back plate plane)")
    ap.add_argument("--wall-brake", type=float, default=0.25,
                    help="virtual wall: braking time [s]; after the step test ~2.5*(L+tau)")
    ap.add_argument("--no-wall", action="store_true", help="disable the virtual wall (NOT in the in-air test)")
    ap.add_argument("--filter-json", default=None,
                    help="action->twist filter from the step test (instead of robosuite coefficients)")
    ap.add_argument("--step-test", action="store_true",
                    help="instead of the policy: twist pulses on each axis, response measurement, new filter")
    ap.add_argument("--step-amp-lin", type=float, default=0.05, help="[step-test] linear amplitude [m/s]")
    ap.add_argument("--step-amp-ang", type=float, default=0.2, help="[step-test] angular amplitude [rad/s]")
    ap.add_argument("--step-on", type=float, default=0.8, help="[step-test] pulse duration [s]")
    ap.add_argument("--step-out", default="step_test",
                    help="[step-test] output prefix: <prefix>_raw.csv, <prefix>_filter.json")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    print("[bridge] loading the core (MuJoCo, arm_d scene) and the policy...")
    core = ObsCore()
    model = PPO.load(SCENE_ARM_D["model"], device="cpu")
    if args.filter_json:
        fa, fb = ST.load_filter(args.filter_json)
        to_twist = ActionToTwist(alpha=fa, beta=fb, speed_scale=args.speed_scale)
        print(f"[bridge] filter from {args.filter_json}: alpha {np.round(fa, 3)}, beta {np.round(fb, 4)}")
    else:
        to_twist = ActionToTwist(speed_scale=args.speed_scale)
    board_p, board_R = parse_board(args.board, core)
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
    node = Bridge(args)
    try:
        if not node.wait_ready():
            node.get_logger().error(f"no /joint_states or TF {node.prof['base']}->{node.prof['flange']}")
            return 1
        if args.backend in ("servo", "ik"):
            ok_s, msg_s = node.start_servo(srv="stop_servo")
            print(f"[bridge] stop_servo before homing: success={ok_s} '{msg_s}'")
        switched = False
        if args.backend == "kortex" and not args.no_switch:
            print(f"[controllers] before: {node.active_controllers()}")
            ok_c, msg_c = node.use_controller("jtc")
            print(f"[controllers] for homing: {msg_c}")
            if not ok_c:
                node.get_logger().error("could not activate the JTC - stop"); return 1
            switched = True
        if args.home:
            if args.home_safe:
                node.spin_for(0.5)
                wps, rap = core.safe_home_path(node.q)
                print(f"[homing] from {np.round(node.q, 3)}: {rap}")
                if wps is None:
                    node.get_logger().error("homing aborted - sending nothing"); return 1
                for wp in wps[:-1]:
                    node.home(wp, args.home_time, quiet=True)
            if not node.home(SCENE_ARM_D["init_qpos"], args.home_time):
                return 1
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
                q0 = JointUnwrapper(SCENE_ARM_D["init_qpos"]).reset(node.q)
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
        unw = JointUnwrapper(SCENE_ARM_D["init_qpos"])
        ok, rep = startup_check(node, core, unw, board_p, board_R)
        print("[FK check]\n  " + rep)
        if not ok and not args.skip_check:
            node.get_logger().error("core and robot DISAGREE on kinematics - stop")
            return 1

        if args.backend == "servo":
            ok_s, msg_s = node.start_servo()
            print(f"[bridge] start_servo: success={ok_s} '{msg_s}'")
        if switched:
            ok_c, msg_c = node.use_controller("twist")
            print(f"[controllers] for driving: {msg_c}")
            if not ok_c:
                node.get_logger().error("could not activate twist_controller - stop"); return 1
        if args.backend not in ("dry", "ik"):
            node.spin_for(1.0)
            if args.home:
                drift = np.abs(np.angle(np.exp(1j * (node.q - np.asarray(SCENE_ARM_D["init_qpos"])))))
                print(f"[bridge] drift after output start [rad]: {np.round(drift, 3)}")
                if drift.max() > 0.03:
                    node.get_logger().error("the arm left the start pose before the loop - stop")
                    return 1
            n_sub = node.tw_pub.get_subscription_count()
            print(f"[bridge] {node.prof['twist']}: subscribers {n_sub}"
                  + ("" if args.backend != "servo" else
                     f", /servo_node/status = {node.servo_status} "
                     f"({SERVO_STATUS.get(node.servo_status, 'no message - is Servo running?')})"))
            if n_sub == 0:
                node.get_logger().error("nobody listens to the twist - Servo/twist_controller not active")
                return 1
            node.watchdog(args.watchdog)
        wall = None
        if not args.no_wall:
            wall = wall_for(core, board_p, board_R, x_wall=args.wall_x, t_brake=args.wall_brake)
            print(f"[wall] grip point no deeper than {1000 * -args.wall_x:.0f} mm before the board back "
                  f"plate; box +-{1000 * wall.half:.0f} mm; no limit on the Spot side (the body is physical)")
        period = 1.0 / args.rate

        def now():
            return node.sim_t if node.sim_t is not None else time.time()

        def output(v, w, q):
            """Shared output of the policy loop and the step test. Returns a status."""
            if args.backend == "ik":
                # velocity -> Jacobian joint step -> JointTrajectory, on the /joint_states
                # branch so the controller does not spin 2*pi
                dq = core.joint_step(q, v, w, period)
                if args.ik_ref:
                    # Integrate the reference pose in the bridge: commanding from the
                    # measured pose realised ~14 % (15.10). |q_ref - q| is capped for contact.
                    q_meas = np.asarray(node.q, dtype=float)
                    if getattr(node, "q_ref", None) is None:
                        node.q_ref = q_meas.copy()
                    node.q_ref = node.q_ref + dq
                    node.q_ref = q_meas + np.clip(node.q_ref - q_meas, -args.ik_ref_max, args.ik_ref_max)
                    node.send_joint_stream(node.q_ref - dq, dq, period, max(1, args.ik_stream))
                elif args.ik_stream > 0:
                    node.send_joint_stream(node.q, dq * args.ik_lead, period, args.ik_stream)
                else:
                    node.send_joint_step(np.asarray(node.q) + dq * args.ik_lead, args.ik_horizon,
                                         dq / period if args.ik_vel else None)
                return f"IK |dq| {np.max(np.abs(dq)):.3f}"
            node.send_twist(v, w)
            return SERVO_STATUS.get(node.servo_status, '-') if args.backend == "servo" else ""

        if args.step_test:
            return run_step_test(node, core, args, output, wall, now, period, unw)

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
            v, w = to_twist.step(a)
            ik_code = ""
            if wall is not None:
                grip = np.asarray(obs["robot0_eef_pos"]) - core.base_in_world
                v, stop_w, d_w = wall.filter(grip, v)
                if stop_w:
                    node.send_zero()
                    node.get_logger().error(f"[wall] {d_w} - end of run")
                    break
                ik_code = d_w
            ik_code = (ik_code + " " + output(v, w, q)).strip()
            dist = float(obs["target_distance"][0])
            align = float(np.degrees(obs["eef_align_with_target"][0]))
            ev = op.poll() if op else None
            out = ik_code
            if k % 5 == 0 or k == args.steps - 1:
                print(f"{k:4d} {tgt:4d} {dist:8.3f} {align:6.1f}° {np.abs(a[:6]).mean():6.2f} "
                      f"{100 * np.mean(np.abs(a[:6]) > 0.99):5.0f}% {np.linalg.norm(v):6.3f} "
                      f"{np.linalg.norm(w):6.3f} {age:7.0f}ms  {out}")
            if log:
                log.writerow([f"{t0 - t_start:.4f}", k, tgt, *np.round(q, 5), *np.round(a[:6], 4),
                              *np.round(v, 4), *np.round(w, 4), f"{dist:.4f}", f"{align:.2f}", f"{age:.1f}", ik_code, ev[0] if ev else ""])
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
            elif hold >= args.reach_hold or k_tgt >= args.max_steps_per_target:
                ok_t = hold >= args.reach_hold
                reached.append((tgt, ok_t, k_tgt))
                print(f"[supervisor] btn{tgt}: {'REACHED' if ok_t else 'STEP LIMIT'} after {k_tgt} steps "
                      f"({k_tgt / args.rate:.1f} s)")
                si, hold, k_tgt = si + 1, 0, 0
                if si >= len(seq):
                    break
            node.wait_until(t_sim_next, t_wall_next)
            t_sim_next += period
            t_wall_next += period
        if len(seq) > 1 or reached:
            print(f"[supervisor] result: {sum(1 for _, ok_t, _ in reached if ok_t)}/{len(seq)} targets "
                  f"{'pressed (operator)' if op else 'reached'}: "
                  + ", ".join(f"btn{t} {'OK' if o else '--'} {n} st." for t, o, n in reached))
    except KeyboardInterrupt:
        print("\n[bridge] interrupted")
    finally:
        node._stop = True
        for _ in range(5):          # twist_controller holds the last command
            node.send_zero()
            time.sleep(0.01)
        if locals().get("switched"):
            ok_c, msg_c = node.use_controller("jtc")   # JTC holds the current pose on activation
            print(f"[controllers] after the run: {msg_c}")
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

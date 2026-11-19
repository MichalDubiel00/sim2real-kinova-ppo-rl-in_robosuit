"""
Common ROS side of every robot backend: /joint_states, TF, homing through the JTC,
heartbeat, watchdog. A backend adds only its output (how (v, w) reaches the arm)
and its start/stop hooks.
"""

import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_msgs.msg import Empty
from rosgraph_msgs.msg import Clock
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener

from core.obs_core import _quat2mat

JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
GRIP_OFFSET = 0.1448      # robosuite grip point along flange Z (6.26)

# Webots world (servo, ik, dry); kinova_vision defaults on hardware - VERIFY (ros2 topic list)
PROFILE_WEBOTS = dict(base="gen3_base_link", flange="end_effector_link",
                      jtc="/kinova_joint_trajectory_controller/joint_trajectory",
                      img="/Gen3/kinova_color/image_color", info="/Gen3/kinova_color/camera_info")


class RobotBase(Node):
    profile = PROFILE_WEBOTS
    # True: output is a held twist -> drift check, subscriber check and watchdog before the loop
    streams_twist = False

    def __init__(self, args):
        super().__init__("arm_d_bridge")
        self.args = args
        self.prof = self.profile
        self.q = self.qd = None
        self.js_stamp = None
        self.create_subscription(JointState, "/joint_states", self._js, 10)
        self.tfb = Buffer()
        self.tfl = TransformListener(self.tfb, self)
        # labpy homes through saved actions and has no JTC topic
        self.jtc_pub = (self.create_publisher(JointTrajectory, self.prof["jtc"], 10)
                        if "jtc" in self.prof else None)
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

    # ------------------------------------------------- backend hooks --
    def prepare_homing(self):
        """Before homing. False = stop."""
        return True

    def start_output(self):
        """After the FK check, before the loop. False = stop."""
        return True

    def output(self, v, w, q):
        """Send the grip-point twist (base frame). Returns a status string."""
        return ""

    def output_status(self):
        """Extra text for the subscriber line before the loop."""
        return ""

    def release(self):
        """In finally, after the zero twists."""

    def _send_zero_locked(self):
        pass

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

    def base_from(self, frame):
        """(p, R) of base <- frame from TF, or None."""
        try:
            tr = self.tfb.lookup_transform(self.prof["base"], frame, rclpy.time.Time())
        except Exception:
            return None
        t, r = tr.transform.translation, tr.transform.rotation
        return np.array([t.x, t.y, t.z]), _quat2mat(np.array([r.x, r.y, r.z, r.w]))

    # -------------------------------------------------------- output --
    def send_zero(self):
        with self._lock:
            self._send_zero_locked()

    def _sent(self):
        self._last_cmd_t = time.time()
        self.hb_pub.publish(Empty())

    def stop_all(self):
        """Zeros (twist_controller holds the last command), then the backend's release."""
        self._stop = True
        for _ in range(5):
            self.send_zero()
            time.sleep(0.01)
        self.release()

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

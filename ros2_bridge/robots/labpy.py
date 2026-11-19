"""
Kinova Gen3 through the lab driver kortex_ros2_python (docs 6.77).

Driver facts (code read 12.11, measured 18.11): TwistStamped on /twist_controller/commands in the
TOOL frame, values halved by the driver, a message with every component < 0.05 is a stop,
Stop() 0.2 s after the last message, Cartesian Joystick caps 0.2 m/s and 20 deg/s apply.
No base frame in TF: flange pose from the training FK. Start pose through saved actions
(/goto_position), route nuc -> Home -> rl_via -> rl_start (Home -> rl_start directly puts the
fingers into Spot's front); the PH scene starts at Home.
"""

import time

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String, Bool

from core.obs_core import wrap_pi
from core.twist_out import base_to_tool, shift_point
from robots.base import RobotBase

DRIVER_SCALE = 0.5           # driver multiplies linear and angular values by 0.5
DEAD_BAND = 0.05             # per component of the message
WEBAPP_TCP_Z = 0.20          # lab Web App tool: TCP 0.20 m along the flange Z, no rotation

# Saved actions [deg, Web App], measured/saved 12.11 and 18.11.
POSES_DEG = {
    "nuc": [0.0, 254.5, 212.2, 0.0, 298.8, 90.0],
    "Home": [0.0, 15.0, 230.0, 0.0, 55.0, 90.0],
    "rl_via": [0.0, 15.0, 231.6, 179.9, 244.3, 180.0],
    "rl_start": [0.0, 84.2, 231.6, 179.9, 244.3, 180.0],
}
ROUTE = ["nuc", "Home", "rl_via", "rl_start"]

PROFILE_LAB = dict(twist="/twist_controller/commands", goto="/goto_position",
                   goto_result="/goto_position/result",
                   img="/kinova_color", info="/kinova_color/camera_info")


def pose_error_deg(q, name):
    return np.degrees(np.abs(wrap_pi(np.asarray(q) - np.radians(POSES_DEG[name]))))


class LabPyRobot(RobotBase):
    profile = PROFILE_LAB
    streams_twist = True
    fk_only = True               # no TF base frame: FK check replaced by a start-pose check

    def __init__(self, args, core):
        super().__init__(args)
        self.core = core
        self.send = not args.no_send
        self.tw_pub = self.create_publisher(TwistStamped, self.prof["twist"], 10)
        self.goto_pub = self.create_publisher(String, self.prof["goto"], 1)
        self.goto_result = None
        self.create_subscription(Bool, self.prof["goto_result"],
                                 lambda m: setattr(self, "goto_result", bool(m.data)), 1)
        self.n_deadband = 0

    # ------------------------------------------------------- state --
    def wait_ready(self, timeout=15.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.q is not None:
                return True
        return False

    def flange_pose(self):
        """(p, quat xyzw) from the training FK; replaces TF (not published by the lab)."""
        if self.q is None:
            return None
        p, R = self.core.flange_pose_R(self.q)
        return p, _mat2quat(R)

    def nearest_pose(self):
        errs = {n: float(pose_error_deg(self.q, n).max()) for n in POSES_DEG}
        name = min(errs, key=errs.get)
        return name, errs[name]

    # ----------------------------------------------------- homing --
    def goto(self, name, timeout=40.0):
        """Run a saved action and wait for /goto_position/result."""
        self.goto_result = None
        t0 = time.time()
        while self.goto_pub.get_subscription_count() == 0 and time.time() - t0 < 3.0:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.goto_pub.get_subscription_count() == 0:
            self.get_logger().error(f"nobody subscribes to {self.prof['goto']} (lab driver not running?)")
            return False
        self.goto_pub.publish(String(data=name))
        while self.goto_result is None and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
        self.spin_for(0.3)
        err = pose_error_deg(self.q, name).max()
        ok = bool(self.goto_result) and err < 1.0
        print(f"[route] {name}: result {self.goto_result}, error {err:.2f} deg after {time.time() - t0:.1f} s")
        return ok

    def route_to(self, target):
        """Follow ROUTE from the nearest saved pose to `target` (either direction)."""
        here, err = self.nearest_pose()
        if err > 3.0:
            self.get_logger().error(f"arm is {err:.1f} deg from the nearest saved pose ({here}); "
                                    f"move it to one of {ROUTE} first")
            return False
        i, j = ROUTE.index(here), ROUTE.index(target)
        legs = ROUTE[i + 1:j + 1] if j > i else ROUTE[j:i][::-1]
        print(f"[route] at {here} ({err:.1f} deg), legs: {legs or 'none'}")
        if not self.send:
            print("[route] --no-send: not moving")
            return err < 3.0 and here == target
        for leg in legs:
            if not self.goto(leg):
                return False
        return True

    def home(self, qpos, duration=None, quiet=False):
        """Start pose = the saved action equal to the scene's init_qpos (rl_start or Home)."""
        errs = {n: float(pose_error_deg(qpos, n).max()) for n in POSES_DEG}
        name = min(errs, key=errs.get)
        assert errs[name] < 0.5, f"scene init_qpos is not a saved action (nearest {name}, {errs[name]:.2f} deg)"
        return self.route_to(name)

    # ----------------------------------------------------- output --
    def output(self, v_base, w_base, q):
        """v_base/w_base refer to the grip point in the base frame."""
        with self._lock:
            _, R = self.core.flange_pose_R(q)
            z = R[:, 2]
            # driver controls the Web App TCP, the policy the grip point
            v_base, w_base = shift_point(v_base, w_base, (WEBAPP_TCP_Z - self.core.grip_offset) * z)
            v_t, w_t = base_to_tool(v_base, w_base, R)
            msg = np.r_[v_t, w_t] / DRIVER_SCALE
            db = bool(np.all(np.abs(msg) < DEAD_BAND) and np.any(msg != 0))
            self.n_deadband += int(db)
            if self.send:
                self._publish(msg)
            self._sent()
        return ("deadband " if db else "") + ("" if self.send else "dry")

    def output_status(self):
        return "" if self.send else "  (--no-send: computed, not published)"

    def _publish(self, msg6):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "tool"
        (m.twist.linear.x, m.twist.linear.y, m.twist.linear.z,
         m.twist.angular.x, m.twist.angular.y, m.twist.angular.z) = map(float, msg6)
        self.tw_pub.publish(m)

    def _send_zero_locked(self):
        # all-zero message = the driver's stop path (zero twist, no 0.2 s coast)
        if self.send:
            self._publish(np.zeros(6))


def _mat2quat(R):
    """xyzw from a rotation matrix (Shepperd)."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = np.zeros(4)
    q[i] = s / 4
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q

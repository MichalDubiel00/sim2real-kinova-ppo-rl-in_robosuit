"""
Webots: MoveIt Servo, TwistStamped in the base frame. Docs 6.53, 6.59.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int8
from std_srvs.srv import Trigger

from core.obs_core import _quat2mat
from core.twist_out import shift_point
from robots.base import RobotBase, GRIP_OFFSET

# moveit_servo (Humble) StatusCode
SERVO_STATUS = {0: "OK", 1: "slowing: singularity", 2: "STOP: singularity",
                3: "slowing: collision", 4: "STOP: collision", 5: "STOP: joint limit"}
TWIST_TOPIC = "/servo_node/delta_twist_cmds"


def start_servo(node, timeout=5.0, srv="start_servo"):
    """Call /servo_node/start_servo (or stop_servo). Without start Servo does
    nothing. start on a running Servo pulls the arm back to its stale pose
    (15.10), so the bridge does stop -> home -> start."""
    cli = node.create_client(Trigger, f"/servo_node/{srv}")
    if not cli.wait_for_service(timeout_sec=timeout):
        return False, f"service /servo_node/{srv} unavailable"
    fut = cli.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
    r = fut.result()
    if r is None:
        return False, "no response"
    return bool(r.success), r.message


class ServoRobot(RobotBase):
    streams_twist = True

    def __init__(self, args):
        super().__init__(args)
        self.prof = dict(self.prof, twist=TWIST_TOPIC)
        self.tw_pub = self.create_publisher(TwistStamped, TWIST_TOPIC, 10)
        self.servo_status = None
        self.create_subscription(Int8, "/servo_node/status",
                                 lambda m: setattr(self, "servo_status", int(m.data)), 10)

    def prepare_homing(self):
        ok, msg = start_servo(self, srv="stop_servo")
        print(f"[bridge] stop_servo before homing: success={ok} '{msg}'")
        return True

    def start_output(self):
        ok, msg = start_servo(self)
        print(f"[bridge] start_servo: success={ok} '{msg}'")
        return True

    def output_status(self):
        return (f", /servo_node/status = {self.servo_status} "
                f"({SERVO_STATUS.get(self.servo_status, 'no message - is Servo running?')})")

    def output(self, v_base, w_base, q):
        with self._lock:
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
            self.tw_pub.publish(m)
            self._sent()
        return SERVO_STATUS.get(self.servo_status, "-")

    def _send_zero_locked(self):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.prof["base"]
        self.tw_pub.publish(m)

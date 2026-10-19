#!/usr/bin/env python3
"""
Deadman: independent guard for twist_controller. Docs 6.62.

twist_controller holds the last command, and the bridge's own watchdog dies with
the bridge (kill -9, OOM, segfault). This process watches /arm_d_bridge/heartbeat;
after --timeout of silence it sends zero twist for --zero-time s and switches to
joint_trajectory_controller. It ignores the twist topic itself, where its own
zeros would look like a live bridge.

Start BEFORE the bridge, in its own terminal:
  ~/miniconda3/envs/robosuite/bin/python ros2_bridge/deadman.py
"""

import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist


class Deadman(Node):
    def __init__(self, a):
        super().__init__("arm_d_deadman")
        self.a = a
        self.last = None
        self.create_subscription(Empty, "/arm_d_bridge/heartbeat", self._hb, 10)
        self.pub = self.create_publisher(Twist, a.twist_topic, 10)
        self.tripped = False

    def _hb(self, _):
        self.last = time.time()
        if self.tripped:
            self.get_logger().info("bridge sending commands again - deadman armed")
            self.tripped = False

    def stop_arm(self):
        t0 = time.time()
        while time.time() - t0 < self.a.zero_time:
            self.pub.publish(Twist())
            rclpy.spin_once(self, timeout_sec=0.01)
        try:
            from controller_manager_msgs.srv import SwitchController
            cli = self.create_client(SwitchController, "/controller_manager/switch_controller")
            if cli.wait_for_service(timeout_sec=1.0):
                req = SwitchController.Request()
                req.activate_controllers = [self.a.jtc_name]
                req.deactivate_controllers = [self.a.twist_name]
                req.strictness = SwitchController.Request.BEST_EFFORT
                req.activate_asap = True
                fut = cli.call_async(req)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
                ok = fut.result() is not None and fut.result().ok
                self.get_logger().warn(f"switched to {self.a.jtc_name}: {ok}")
        except Exception as e:                     # zeros were already sent
            self.get_logger().error(f"controller switch failed: {e!s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=0.15, help="[s] without heartbeat -> stop")
    ap.add_argument("--zero-time", type=float, default=0.5, help="[s] how long to send zeros")
    ap.add_argument("--twist-topic", default="/twist_controller/commands")
    ap.add_argument("--jtc-name", default="joint_trajectory_controller")
    ap.add_argument("--twist-name", default="twist_controller")
    a = ap.parse_args()
    rclpy.init()
    n = Deadman(a)
    print(f"[deadman] watching: no /arm_d_bridge/heartbeat for {a.timeout} s -> zero + {a.jtc_name}",
          flush=True)
    try:
        while rclpy.ok():
            rclpy.spin_once(n, timeout_sec=0.01)
            if n.last is not None and not n.tripped and time.time() - n.last > a.timeout:
                n.get_logger().error(f"bridge silent for {1000 * (time.time() - n.last):.0f} ms - STOP")
                n.tripped = True
                n.stop_arm()
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

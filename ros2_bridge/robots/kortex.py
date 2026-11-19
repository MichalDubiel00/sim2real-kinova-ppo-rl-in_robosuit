"""
Kinova Gen3 hardware via ros2_kortex: twist_controller, Twist in the TOOL frame.
JTC for homing, twist_controller for the policy, back to the JTC on exit.
"""

import time

import numpy as np
from geometry_msgs.msg import Twist

from core.obs_core import _quat2mat
from core.twist_out import base_to_tool, shift_point
from robots.base import RobotBase, GRIP_OFFSET

PROFILE_KORTEX = dict(base="base_link", flange="end_effector_link",
                      cm="/controller_manager", jtc_name="joint_trajectory_controller",
                      twist_name="twist_controller",
                      jtc="/joint_trajectory_controller/joint_trajectory",
                      twist="/twist_controller/commands",
                      img="/camera/color/image_raw", info="/camera/color/camera_info")


class KortexRobot(RobotBase):
    profile = PROFILE_KORTEX
    streams_twist = True

    def __init__(self, args):
        super().__init__(args)
        self.tw_pub = self.create_publisher(Twist, self.prof["twist"], 10)
        self.switched = False

    # ----------------------------------------------------- controllers --
    def _cm_call(self, srv_type, name, req, timeout=5.0):
        import rclpy
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
        from controller_manager_msgs.srv import SwitchController
        jtc, tw = self.prof["jtc_name"], self.prof["twist_name"]
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

    # ------------------------------------------------------------ hooks --
    def prepare_homing(self):
        if self.args.no_switch:
            return True
        print(f"[controllers] before: {self.active_controllers()}")
        ok, msg = self.use_controller("jtc")
        print(f"[controllers] for homing: {msg}")
        if not ok:
            self.get_logger().error("could not activate the JTC - stop")
            return False
        self.switched = True
        return True

    def start_output(self):
        if not self.switched:
            return True
        ok, msg = self.use_controller("twist")
        print(f"[controllers] for driving: {msg}")
        if not ok:
            self.get_logger().error("could not activate twist_controller - stop")
        return ok

    def release(self):
        if self.switched:
            ok, msg = self.use_controller("jtc")   # JTC holds the current pose on activation
            print(f"[controllers] after the run: {msg}")

    def output(self, v_base, w_base, q):
        """v_base/w_base refer to the GRIP POINT in the base frame."""
        with self._lock:
            fp = self.flange_pose()
            if fp is None:
                self._send_zero_locked()
                return ""
            R = _quat2mat(fp[1])           # flange orientation = firmware TOOL frame
            z = R[:, 2]                    # (assumes an unrotated Web App tool - verify)
            # firmware controls the Web App TCP, the policy the grip point
            v_base, w_base = shift_point(v_base, w_base, (self.args.tool_tcp_z - GRIP_OFFSET) * z)
            v_t, w_t = base_to_tool(v_base, w_base, R)
            m = Twist()
            (m.linear.x, m.linear.y, m.linear.z) = map(float, v_t)
            (m.angular.x, m.angular.y, m.angular.z) = map(float, w_t)
            self.tw_pub.publish(m)
            self._sent()
        return ""

    def _send_zero_locked(self):
        self.tw_pub.publish(Twist())

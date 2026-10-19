#!/usr/bin/env python3
"""
Mock of ros2_kortex for testing the bridge without the robot. Docs 6.62.

Mimics what the bridge sees from a Gen3: wrapped /joint_states, TF
base_link -> end_effector_link, twist_controller (TOOL-frame Twist, holds the last
command; --dead / --tau / --gain), JTC homing, list/switch_controller.
Kinematics come from the training model (ObsCore), so this tests bridge logic,
not the real robot's kinematics.

  ~/miniconda3/envs/robosuite/bin/python ros2_bridge/mock_kortex.py --dead 0.04 --tau 0.06 --gain 0.95
"""

import sys
sys.modules["tensorflow"] = None

import argparse
import collections
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, TransformStamped
from trajectory_msgs.msg import JointTrajectory
from tf2_ros import TransformBroadcaster
from controller_manager_msgs.srv import ListControllers, SwitchController
from controller_manager_msgs.msg import ControllerState

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obs_core import ObsCore, SCENE_ARM_D, _quat2mat, wrap_pi

JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
GRIP_OFFSET = 0.1448
# Training-model joint limits (kinova3_6dof/robot.xml); j1, j4, j6 continuous.
Q_LO = np.array([-np.inf, -2.25, -2.58, -np.inf, -2.10, -np.inf])
Q_HI = np.array([np.inf, 2.25, 2.58, np.inf, 2.10, np.inf])
QUAT_SITE_REL = np.array([0.0, 0.0, -0.70710678, 0.70710678])


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


class MockKortex(Node):
    def __init__(self, a):
        super().__init__("mock_kortex")
        self.a = a
        self.core = ObsCore()
        self.q = np.array(a.start, dtype=np.float64)
        self.qd = np.zeros(6)
        self.active = {"joint_trajectory_controller": "active", "twist_controller": "inactive"}
        self.twist_cmd = np.zeros(6)                 # last command, held
        self.delay = collections.deque()             # (t, cmd) queue for dead time
        self.v_state = np.zeros(6)                   # grip-point velocity after the lag
        self.traj = None                             # (t0, q0, q1, T)
        self.js_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.tfb = TransformBroadcaster(self)
        self.create_subscription(Twist, "/twist_controller/commands", self._tw, 10)
        self.create_subscription(JointTrajectory, "/joint_trajectory_controller/joint_trajectory", self._jt, 10)
        self.create_service(ListControllers, "/controller_manager/list_controllers", self._list)
        self.create_service(SwitchController, "/controller_manager/switch_controller", self._switch)
        self.n_tw = 0

    # ------------------------------------------------------------- uslugi --
    def _list(self, req, res):
        for n, s in self.active.items():
            c = ControllerState(); c.name = n; c.state = s
            res.controller.append(c)
        return res

    def _switch(self, req, res):
        for n in req.deactivate_controllers:
            self.active[n] = "inactive"
        for n in req.activate_controllers:
            self.active[n] = "active"
        res.ok = sum(s == "active" for s in self.active.values()) == 1
        if self.active["joint_trajectory_controller"] == "active":
            self.traj = None                         # JTC holds the current pose on activation
        self.get_logger().info(f"switch -> {self.active}")
        return res

    # ------------------------------------------------------------ komendy --
    def _tw(self, m):
        self.n_tw += 1
        if self.active["twist_controller"] != "active":
            return
        self.twist_cmd = np.array([m.linear.x, m.linear.y, m.linear.z, m.angular.x, m.angular.y, m.angular.z])

    def _jt(self, m):
        if self.active["joint_trajectory_controller"] != "active" or not m.points:
            return
        pt = m.points[-1]
        T = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
        q1 = np.array([pt.positions[m.joint_names.index(j)] for j in JOINTS])
        # nearest branch, like the driver
        q1 = self.q + wrap_pi(q1 - self.q)
        self.traj = (time.time(), self.q.copy(), q1, max(T, 0.05))

    # -------------------------------------------------------------- fizyka --
    def flange(self):
        _, obs = self.core.build(self.q, np.zeros(6), *self.core.nominal_board_pose(), 0)
        grip = np.asarray(obs["robot0_eef_pos"]) - self.core.base_in_world
        q_fl = quat_mul(np.asarray(obs["robot0_eef_quat"]), QUAT_SITE_REL * np.array([-1, -1, -1, 1]))
        R = _quat2mat(q_fl)
        return grip - GRIP_OFFSET * R[:, 2], q_fl, R, grip

    def step(self, dt):
        now = time.time()
        if self.active["twist_controller"] == "active":
            self.delay.append((now, self.twist_cmd.copy()))
            while self.delay and now - self.delay[0][0] >= self.a.dead:
                self._last_delayed = self.delay.popleft()[1]
            cmd = getattr(self, "_last_delayed", np.zeros(6))   # command from L seconds ago
            fl, _, R, grip = self.flange()
            v_tcp, w = R @ cmd[:3], R @ cmd[3:]                   # TOOL -> base
            tcp = fl + self.a.tcp_z * R[:, 2]
            v_grip = v_tcp + np.cross(w, grip - tcp)
            target = self.a.gain * np.r_[v_grip, w]
            self.v_state += dt / self.a.tau * (target - self.v_state)
            # small damping: 0.02 left ~34 mm/s translation error in large rotations
            dq = self.core.joint_step(self.q, self.v_state[:3], self.v_state[3:], dt, damping=1e-3)
            q_new = np.clip(self.q + dq, Q_LO, Q_HI)          # joint stops at its limit
            self.qd = (q_new - self.q) / dt
            self.q = q_new
        elif self.traj is not None:
            t0, q0, q1, T = self.traj
            s = min((now - t0) / T, 1.0)
            q_new = q0 + s * (q1 - q0)
            self.qd = (q_new - self.q) / dt
            self.q = q_new
            self.v_state[:] = 0.0
        else:
            self.qd[:] = 0.0
            self.v_state[:] = 0.0

    def publish(self):
        st = self.get_clock().now().to_msg()
        m = JointState(); m.header.stamp = st
        m.name = JOINTS
        m.position = [float(x) for x in wrap_pi(self.q)]          # wrapped like Kortex
        m.velocity = [float(x) for x in self.qd]
        self.js_pub.publish(m)
        fl, q_fl, _, _ = self.flange()
        t = TransformStamped(); t.header.stamp = st
        t.header.frame_id = "base_link"; t.child_frame_id = "end_effector_link"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, fl)
        (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z,
         t.transform.rotation.w) = map(float, q_fl)
        self.tfb.sendTransform(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dead", type=float, default=0.04, help="twist dead time [s]")
    ap.add_argument("--tau", type=float, default=0.06, help="time constant [s]")
    ap.add_argument("--gain", type=float, default=0.95, help="twist gain")
    ap.add_argument("--tcp-z", type=float, default=GRIP_OFFSET, help="tool TCP along flange Z [m]")
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--start", type=lambda s: [float(x) for x in s.split(",")],
                    default=[0.0, 0.9, -2.0, 0.0, -1.2, 1.57], help="start angles (not outside arm_d)")
    ap.add_argument("--seconds", type=float, default=600)
    a = ap.parse_args()
    rclpy.init()
    n = MockKortex(a)
    print(f"[mock] start q {np.round(n.q, 3)}; twist: L {a.dead} s, tau {a.tau} s, K {a.gain}, TCP {a.tcp_z} m",
          flush=True)
    dt, t_end, k = 1.0 / a.rate, time.time() + a.seconds, 0
    t_next = time.time()
    while time.time() < t_end:
        rclpy.spin_once(n, timeout_sec=0.0)
        n.step(dt)
        if True:
            n.publish()
        k += 1
        t_next += dt
        while time.time() < t_next:
            rclpy.spin_once(n, timeout_sec=max(0.0, t_next - time.time()))
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Mock of the lab driver kortex_ros2_python for closed-loop bridge tests without the robot (docs 6.77).

Driver side as the real one: TwistStamped on /twist_controller/commands in the TOOL frame (flange
axes, Web App TCP 0.20 m), values x0.5, a message with all components < 0.05 = stop, Stop() 0.2 s
after the last message; /goto_position (String) runs a saved action at 25 deg/s and answers on
/goto_position/result; /joint_states with velocities, j1/j4/j6 in [0, 2pi).
Twist response as measured 18.11: dead time 60 ms, first order (tau 65 ms + ~20 ms joint loop).

Plant (--plant sim, default): the training MuJoCo scene with the board, E-stop caps and Spot,
at --base-height (stand-off rule as in training); the arm follows the TCP twist through a
computed-torque joint velocity loop, so caps resist and get pressed. The env's own hook applies
the lab caps (0.2 m/s, 20 deg/s, 50 deg/s per joint). Runs on its own clock (/clock): the
bridge paces on it, so CPU load does not change the timing.
--plant kin: kinematic arm in free air (no contacts), wall clock.

  source /opt/ros/humble/setup.bash   # + a LOCAL-ONLY RMW, never the lab router
  ~/miniconda3/envs/robosuite/bin/python ros2_bridge/tools/mock_labpy.py --start nuc --base-height 0.62
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
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String, Bool, Int32
from rosgraph_msgs.msg import Clock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.obs_core import ObsCore, SCENES, NOMINAL_OVERRIDES, wrap_pi
from robots.labpy import POSES_DEG

JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
Q_LO = np.array([-np.inf, -2.25, -2.58, -np.inf, -2.10, -np.inf])
Q_HI = np.array([np.inf, 2.25, 2.58, np.inf, 2.10, np.inf])
TCP_Z = 0.20


class SimPlant:
    """Training scene driven by a joint velocity loop (no OSC, no policy)."""
    sim_clock = True

    def __init__(self, scene, base_height, q0, kv=90.0, kp=2000.0, windup=0.01):
        import mujoco
        import yaml
        from envs.button_board import ButtonBoardEnv
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training"))
        from train_board import env_kwargs_from_cfg
        self.mj = mujoco
        cfg = yaml.safe_load(open(scene["cfg"]))
        kw = env_kwargs_from_cfg(cfg)
        kw.update(NOMINAL_OVERRIDES)
        kw["horizon"] = 10 ** 9
        if base_height is not None:
            kw.update(base_z_range=[base_height, base_height], standoff_keep_reach=True)
        self.env = env = ButtonBoardEnv(**kw)
        env.reset()
        rob = env.robots[0]
        self.m, self.d = env.sim.model._model, env.sim.data._data
        self.qi = np.asarray(rob._ref_joint_pos_indexes)
        self.vi = np.asarray(rob._ref_joint_vel_indexes)
        self.act = np.asarray(rob._ref_arm_joint_actuator_indexes)
        self.hand = env.sim.model.body_name2id("robot0_right_hand")
        self.base = env.sim.data.body_xpos[env.sim.model.body_name2id("robot0_base")].copy()
        self.dt = float(self.m.opt.timestep)
        self.kv, self.kp, self.windup = kv, kp, windup
        # gripper closed as in training: let the env's gripper controller close it, then hold its ctrl
        for _ in range(15):
            env.step(np.zeros(7))
        self.g_act = np.asarray(rob._ref_joint_gripper_actuator_indexes["right"])
        self.g_ctrl = self.d.ctrl[self.g_act].copy()
        self.set_q(q0)
        self.t = 0.0
        self.btn_adr = list(env._btn_qposadr)
        self.pressed = set()
        self.on_press = None

    def set_q(self, q):
        self.d.qpos[self.qi] = q
        self.d.qvel[self.vi] = 0.0
        self.q_ref = np.asarray(q, dtype=np.float64).copy()
        self.mj.mj_forward(self.m, self.d)

    def q(self):
        return self.d.qpos[self.qi].copy()

    def qd(self):
        return self.d.qvel[self.vi].copy()

    def flange(self):
        return self.d.xpos[self.hand].copy(), self.d.xmat[self.hand].reshape(3, 3).copy()

    def step(self, tw_base):
        """One physics substep tracking the TCP twist (base frame = world up to translation)."""
        m, d = self.m, self.d
        p_f, R = self.flange()
        tcp = p_f + TCP_Z * R[:, 2]
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        self.mj.mj_jac(m, d, jp, jr, tcp, self.hand)
        J = np.vstack([jp[:, self.vi], jr[:, self.vi]])
        qd_des = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), tw_base)
        M = np.zeros((m.nv, m.nv))
        self.mj.mj_fullM(m, M, d.qM)
        Ma = M[np.ix_(self.vi, self.vi)]
        # stiff servo: integrated reference (pushes against a cap like a position-tracking joint),
        # bounded so a blocked arm cannot wind up beyond ~kp*windup
        q = d.qpos[self.qi]
        self.q_ref = q + np.clip(self.q_ref + qd_des * self.dt - q, -self.windup, self.windup)
        tau = Ma @ (self.kp * (self.q_ref - q) + self.kv * (qd_des - d.qvel[self.vi])) + d.qfrc_bias[self.vi]
        d.ctrl[self.act] = 0.0
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[self.vi] = tau
        if self.g_act is not None and self.g_ctrl is not None:
            d.ctrl[self.g_act] = self.g_ctrl
        self.env.sim.step()
        self.t += self.dt
        for i, a in enumerate(self.btn_adr):
            if d.qpos[a] > 0.0059 and i not in self.pressed:
                self.pressed.add(i)
                print(f"[mock lab] t={self.t:.1f} s: btn{i} pressed", flush=True)
                if self.on_press:
                    self.on_press(i)

    def hold(self):
        self.step(np.zeros(6))


class KinPlant:
    """Kinematic arm in free air (training model kinematics), wall clock."""
    sim_clock = False

    def __init__(self, scene, base_height, q0):
        import mujoco
        self.mj = mujoco
        self.core = ObsCore(scene)
        sim = self.core.env.sim
        self.m, self.d = sim.model._model, sim.data._data
        self.hand = sim.model.body_name2id("robot0_right_hand")
        self.qi, self.vi = self.core.q_idx, self.core.qd_idx
        self._q = np.asarray(q0, dtype=np.float64).copy()
        self._qd = np.zeros(6)
        self.dt = 0.005
        self.t = 0.0
        self.pressed = set()

    def set_q(self, q):
        self._q = np.asarray(q, dtype=np.float64).copy()
        self._qd[:] = 0

    def q(self):
        return self._q.copy()

    def qd(self):
        return self._qd.copy()

    def step(self, tw_base):
        self.d.qpos[self.qi] = self._q
        self.mj.mj_kinematics(self.m, self.d)
        self.mj.mj_comPos(self.m, self.d)
        R = self.d.xmat[self.hand].reshape(3, 3)
        tcp = self.d.xpos[self.hand] + TCP_Z * R[:, 2]
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        self.mj.mj_jac(self.m, self.d, jp, jr, tcp, self.hand)
        J = np.vstack([jp[:, self.vi], jr[:, self.vi]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), tw_base * self.dt)
        q_new = np.clip(self._q + dq, Q_LO, Q_HI)
        self._qd = (q_new - self._q) / self.dt
        self._q = q_new
        self.t += self.dt

    def flange(self):
        self.d.qpos[self.qi] = self._q
        self.mj.mj_kinematics(self.m, self.d)
        return self.d.xpos[self.hand].copy(), self.d.xmat[self.hand].reshape(3, 3).copy()

    def hold(self):
        self._qd[:] = 0
        self.t += self.dt


class MockLab(Node):
    def __init__(self, a):
        super().__init__("mock_kortex_omni_controller")
        self.a = a
        q0 = wrap_pi(np.radians(POSES_DEG[a.start])) if a.start in POSES_DEG else \
            np.array([float(x) for x in a.start.split(",")])
        Plant = SimPlant if a.plant == "sim" else KinPlant
        self.p = Plant(SCENES[a.scene], a.base_height, q0)
        self.cmd = np.zeros(6)                 # TOOL frame [m/s, rad/s] after the driver scale
        self.last_msg = -1.0
        self.delay = collections.deque()
        self.held = np.zeros(6)
        self.v_state = np.zeros(6)
        self.traj = None
        self.js_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.res_pub = self.create_publisher(Bool, "/goto_position/result", 1)
        self.clk_pub = self.create_publisher(Clock, "/clock", 10) if self.p.sim_clock else None
        self.create_subscription(TwistStamped, "/twist_controller/commands", self._tw, 1)
        self.create_subscription(String, "/goto_position", self._goto, 1)
        self.n_msg = self.n_db = 0
        # press events, as a press detector would publish them (bridge --press-topic)
        self.press_pub = self.create_publisher(Int32, "/estop_pressed", 10)
        self.p.on_press = lambda i: self.press_pub.publish(Int32(data=int(i)))

    def now(self):
        return self.p.t if self.p.sim_clock else time.time()

    def _tw(self, m):
        t = m.twist
        raw = np.array([t.linear.x, t.linear.y, t.linear.z, t.angular.x, t.angular.y, t.angular.z])
        self.n_msg += 1
        if np.all(np.abs(raw) < 0.05):
            self.n_db += int(np.any(raw != 0))
            self.cmd = np.zeros(6)
            return
        self.cmd = raw * 0.5
        self.last_msg = self.now()

    def _goto(self, m):
        if m.data not in POSES_DEG:
            self.res_pub.publish(Bool(data=False)); return
        q = self.p.q()
        q1 = q + wrap_pi(np.radians(POSES_DEG[m.data]) - q)   # shortest path
        T = max(np.degrees(np.abs(q1 - q)).max() / 25.0, 0.5)
        self.traj = (self.now(), q, q1, T)
        self.cmd = np.zeros(6)
        self.get_logger().info(f"goto {m.data}: {T:.1f} s")

    def step(self):
        now = self.now()
        if self.traj is not None:
            t0, q0, q1, T = self.traj
            s = min((now - t0) / T, 1.0)
            self.p.set_q(q0 + s * (q1 - q0))
            self.p.hold() if not self.p.sim_clock else setattr(self.p, "t", self.p.t + self.p.dt)
            if s >= 1.0:
                self.traj = None
                self.res_pub.publish(Bool(data=True))
            return
        if now - self.last_msg > 0.2:
            self.cmd = np.zeros(6)               # driver timeout -> Stop()
        self.delay.append((now, self.cmd.copy()))
        while self.delay and now - self.delay[0][0] >= self.a.dead:
            self.held = self.delay.popleft()[1]
        _, R = self.p.flange()
        v, w = R @ self.held[:3], R @ self.held[3:]          # TOOL -> base
        v *= min(1.0, self.a.v_cap / max(np.linalg.norm(v), 1e-12))
        w *= min(1.0, self.a.w_cap / max(np.linalg.norm(w), 1e-12))
        self.v_state += self.p.dt / self.a.tau * (np.r_[v, w] - self.v_state)
        self.p.step(self.v_state)

    def publish(self):
        st = self.get_clock().now().to_msg()
        if self.clk_pub is not None:
            c = Clock()
            c.clock.sec = int(self.p.t)
            c.clock.nanosec = int((self.p.t % 1) * 1e9)
            self.clk_pub.publish(c)
            st = c.clock
        m = JointState()
        m.header.stamp = st
        m.name = JOINTS
        q = wrap_pi(self.p.q())
        for j in (0, 3, 5):
            q[j] = q[j] % (2 * np.pi)            # lab: continuous joints in [0, 2pi)
        m.position = [float(x) for x in q]
        m.velocity = [float(x) for x in self.p.qd()]
        self.js_pub.publish(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="ps")
    ap.add_argument("--plant", choices=["sim", "kin"], default="sim")
    ap.add_argument("--start", default="nuc", help="saved pose name or 6 angles [rad]")
    ap.add_argument("--base-height", type=float, default=None, help="arm base above the floor [m] (sim plant)")
    ap.add_argument("--dead", type=float, default=0.06)
    ap.add_argument("--tau", type=float, default=0.065)
    ap.add_argument("--v-cap", type=float, default=0.2)
    ap.add_argument("--w-cap", type=float, default=0.349)
    ap.add_argument("--rtf", type=float, default=1.0, help="[sim] max real-time factor (wall pacing)")
    ap.add_argument("--seconds", type=float, default=900, help="plant time limit [s]")
    a = ap.parse_args()
    rclpy.init()
    n = MockLab(a)
    print(f"[mock lab] {a.plant} plant, start {a.start}, base height {a.base_height}; "
          f"L {a.dead} s, tau {a.tau} s, caps {a.v_cap} m/s {a.w_cap} rad/s", flush=True)
    k, t_wall0 = 0, time.time()
    pub_every = max(1, int(round(0.005 / n.p.dt)))
    while n.p.t < a.seconds:
        rclpy.spin_once(n, timeout_sec=0.0)
        n.step()
        k += 1
        if k % pub_every == 0:
            n.publish()
        if k % int(10.0 / n.p.dt) == 0:
            print(f"[mock lab] t={n.p.t:.0f} s (wall {time.time() - t_wall0:.0f} s): {n.n_msg} twists, "
                  f"{n.n_db} in the dead band, pressed {sorted(n.p.pressed)}", flush=True)
        lead = n.p.t / a.rtf - (time.time() - t_wall0)    # never faster than rtf x real time
        if lead > 0:
            time.sleep(lead) if not n.p.sim_clock else rclpy.spin_once(n, timeout_sec=lead)
        if not n.p.sim_clock:
            pass
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

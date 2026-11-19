"""
Webots: twist -> Jacobian joint step (MuJoCo, DLS) -> JointTrajectory streamed to
the JTC. Docs 6.59-6.60.
"""

import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from robots.base import RobotBase, JOINTS
from robots.servo import start_servo


class IkRobot(RobotBase):
    def __init__(self, args, core):
        super().__init__(args)
        self.core = core
        self.period = 1.0 / args.rate
        self.q_ref = None

    def prepare_homing(self):
        ok, msg = start_servo(self, srv="stop_servo")
        print(f"[bridge] stop_servo before homing: success={ok} '{msg}'")
        return True

    def output(self, v, w, q):
        a = self.args
        # on the /joint_states branch so the controller does not spin 2*pi
        dq = self.core.joint_step(q, v, w, self.period)
        if a.ik_ref:
            # Integrate the reference pose in the bridge: commanding from the
            # measured pose realised ~14 % (15.10). |q_ref - q| is capped for contact.
            q_meas = np.asarray(self.q, dtype=float)
            if self.q_ref is None:
                self.q_ref = q_meas.copy()
            self.q_ref = self.q_ref + dq
            self.q_ref = q_meas + np.clip(self.q_ref - q_meas, -a.ik_ref_max, a.ik_ref_max)
            self.send_joint_stream(self.q_ref - dq, dq, max(1, a.ik_stream))
        elif a.ik_stream > 0:
            self.send_joint_stream(self.q, dq * a.ik_lead, a.ik_stream)
        else:
            self.send_joint_step(np.asarray(self.q) + dq * a.ik_lead, a.ik_horizon,
                                 dq / self.period if a.ik_vel else None)
        return f"IK |dq| {np.max(np.abs(dq)):.3f}"

    def send_joint_stream(self, q_now, dq, n_points):
        """Points q + k*dq at k*period, k = 1..n. Each command replaces the
        trajectory before it ends, so the arm never brakes per step (a single
        point realised ~14 % of the step)."""
        msg = JointTrajectory()
        msg.joint_names = JOINTS
        for k in range(1, n_points + 1):
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in np.asarray(q_now) + k * np.asarray(dq)]
            t = k * self.period
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t % 1) * 1e9)
            msg.points.append(pt)
        self.jtc_pub.publish(msg)

    def send_joint_step(self, q_cmd, duration, qd_cmd=None):
        """Single point. qd_cmd sets point velocities; without them every
        replacement restarts from zero."""
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

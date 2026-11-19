#!/usr/bin/env python3
"""
Ground-truth board pose in Webots (no ROS). Docs 6.58. Reference for ArUco and the
source of --board webots.

1. Button tops from the world file: DEF estop + five sockets, top = socket centre
   + 0.075 m along local +Y.
2. Rigid fit (Kabsch) of the training socket layout to those tops -> cross-centre
   pose in the world; the residual is the Webots model mismatch.
3. World -> odom as spot_driver.py publishes it; valid only for a Spot start yaw of
   +-pi (checked on load).
4. The bridge adds odom -> gen3_base_link from TF every step.
"""

import itertools
import re

import numpy as np

HEIGHT = 0.52
SOCKETS = {                    # children of DEF estop: (translation, axis-angle)
    "estop": ((0.16, 0.0400012, 0.04), (1, 0, 0, 1.5708)),
    "estop(4)": ((0.16, -0.0699988, 0.0117153), (1, 0, 0, 2.3562)),
    "estop(3)": ((0.27, 0.0400012, -0.00242654),
                 (-0.8628565621288802, -0.35740681862788, 0.3574058186283875, -1.7177753071795863)),
    "estop(2)": ((0.16, 0.131717, 0.0117156), (-1, 0, 0, -0.7854053071795866)),
    "estop(1)": ((0.0538927, 0.0400012, 0.000502356),
                 (0.8628563066501258, -0.3574071270187627, 0.35740612701840735, 1.71777)),
}
CAP_TOP = 0.075


def R_axang(x, y, z, a):
    k = np.array([x, y, z], dtype=float)
    k /= np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def read_def(world_file, name):
    """(translation, R) of node DEF <name> in a .wbt file."""
    txt = open(world_file).read()
    i = txt.index(f"DEF {name} ")
    blk = txt[i:i + 600]
    t = [float(v) for v in re.search(r"translation\s+(\S+)\s+(\S+)\s+(\S+)", blk).groups()]
    r = [float(v) for v in re.search(r"rotation\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", blk).groups()]
    return np.array(t), R_axang(*r)


def caps_world(p_e, R_e):
    names, P, N = [], [], []
    for n, (t, r) in SOCKETS.items():
        Ri = R_axang(*r)
        names.append(n)
        P.append(p_e + R_e @ (np.array(t) + Ri @ np.array([0, CAP_TOP, 0])))
        N.append(R_e @ Ri @ np.array([0, 1.0, 0]))
    return names, np.array(P), np.array(N)


def fit_layout(btn_offsets, btn_normals, P, N):
    """Kabsch: training offsets (relative to the cross centre) -> Webots points."""
    best = None
    for perm in itertools.permutations(range(len(P))):
        A = btn_offsets[list(perm)]
        ca, cb = A.mean(0), P.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (P - cb))
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        p = cb - R @ ca
        res = np.linalg.norm(A @ R.T + p - P, axis=1)
        if best is None or res.max() < best[0]:
            nerr = np.degrees(np.arccos(np.clip(np.sum((btn_normals[list(perm)] @ R.T) * N, 1), -1, 1)))
            best = (res.max(), perm, R, p, res, nerr)
    return best[1:]


class WebotsBoardTruth:
    def __init__(self, world_file, btn_offsets, btn_normals):
        p_e, R_e = read_def(world_file, "estop")
        self.spot_p0, R_s0 = read_def(world_file, "Spot")
        if not np.allclose(R_s0, Rz(np.pi), atol=1e-3):
            raise ValueError("Spot start rotation != Rz(pi) - the spot_driver.py odom formula "
                             "(negating x,y) does not apply")
        self.names, P, N = caps_world(p_e, R_e)
        self.perm, R_w, p_w, self.res, self.nerr = fit_layout(btn_offsets, btn_normals, P, N)
        self.p_world, self.R_world = p_w, R_w
        Rt = Rz(np.pi).T
        self.p_odom = Rt @ (p_w - self.spot_p0) + np.array([0, 0, HEIGHT + 0.095])
        self.R_odom = Rt @ R_w

    def in_base(self, T_base_odom_p, T_base_odom_R):
        """Board pose in the arm base frame, given TF base <- odom."""
        return (T_base_odom_R @ self.p_odom + T_base_odom_p, T_base_odom_R @ self.R_odom)

    def report(self):
        m = {self.names[i]: f"btn{self.perm[i]}" for i in range(len(self.names))}
        return (f"assignment {m}\n  residual [mm] {np.round(1000 * self.res, 1)}, "
                f"normals [deg] {np.round(self.nerr, 1)}\n"
                f"  cross centre: world {np.round(self.p_world, 4)}, odom {np.round(self.p_odom, 4)}")

    def stand_pose_world(self, board_in_base_nominal, base_link_to_arm=(0.256, 0.0, 0.112)):
        """Where to put Spot's base_link (world, yaw) so the board sits relative
        to the arm base as in training (board R = I)."""
        R_base = self.R_world                      # arm base frame in the world
        arm = self.p_world - R_base @ np.asarray(board_in_base_nominal)
        bl = arm - R_base @ np.asarray(base_link_to_arm)
        yaw = np.arctan2(R_base[1, 0], R_base[0, 0])
        return bl, yaw

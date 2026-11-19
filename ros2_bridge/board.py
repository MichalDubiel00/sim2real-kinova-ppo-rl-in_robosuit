"""
Board pose sources for --board: nominal | x,y,z[,rx,ry,rz] | aruco | webots.
"""

import time

import numpy as np
import rclpy

from core.obs_core import _quat2mat
import core.aruco_board as AB


def axang_to_R(v):
    v = np.asarray(v, dtype=np.float64)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


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

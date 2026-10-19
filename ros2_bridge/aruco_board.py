#!/usr/bin/env python3
"""
Board pose from ArUco (Gen3 wrist camera). Docs 6.59.

T_base_board = T_base_flange (TF) * T_flange_cam * T_cam_marker (PnP) * inv(T_board_marker)
  T_cam_marker    cv2.aruco + PnP; OpenCV marker frame (x right, y up, z to camera)
  T_flange_cam    camera extrinsic, optical frame (z forward, x right, y down).
                  Webots: from the proto (the URDF TF is off by ~2.4 cm).
                  Hardware: Kinova Vision module calibration.
  T_board_marker  marker placement in the training board frame. MEASURE on hardware.
Measured once, on a still arm, averaged over N frames: the board pose is constant
per episode, as in training.
"""

import numpy as np

try:
    import cv2
except ImportError:          # allows logic tests without OpenCV
    cv2 = None

DICT = "DICT_4X4_50"
MARKER_ID = 0
MARKER_SIZE = 0.048                   # black square side [m] (384/512 of 0.064)

# Marker in the board frame (training convention); R columns = marker axes.
# Seen from the robot (+x board, +z up): right = -y board.
P_BOARD_MARKER = np.array([-0.005, 0.0, 0.12])
R_BOARD_MARKER = np.array([[0.0, 0.0, -1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])

# Webots: kinova_color camera relative to end_effector_link, from the
# Gen36dofWithRobotiq2f140 proto + spot.wbt (15.10). R columns = optical axes.
P_FLANGE_CAM_WEBOTS = np.array([0.02, 0.07, 0.01])
# Optical frame = flange rotated 180 deg about Z (viewing axis = tool axis).
R_FLANGE_CAM_WEBOTS = np.diag([-1.0, -1.0, 1.0])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def inv(p, R):
    return -R.T @ p, R.T


def compose(a, b):
    """(p,R) a * b"""
    return a[0] + a[1] @ b[0], a[1] @ b[1]


def detect(img_bgr, K, D, marker_id=MARKER_ID, size=MARKER_SIZE, dict_name=DICT):
    """Returns (candidates, corners[4,2]) or None. Candidates = [(p, R, err_px)].
    A planar marker seen nearly head-on has two PnP solutions with almost equal
    error (IPPE picked the wrong one, 28 deg); choose_candidate() decides."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name)), params)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or marker_id not in ids.ravel():
        return None
    c = corners[list(ids.ravel()).index(marker_id)].reshape(4, 2).astype(np.float64)
    return pnp_candidates(c, K, D, size), c


def marker_obj(size=MARKER_SIZE):
    h = size / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def pnp_candidates(corners, K, D, size=MARKER_SIZE):
    obj = marker_obj(size)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, corners, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    rvecs, tvecs = list(rvecs), list(tvecs)
    # IPPE gave 0.8 px / 28 deg error on noise-free data, ITERATIVE was exact: use both
    ok, rv_i, tv_i = cv2.solvePnP(obj, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        rvecs.append(rv_i); tvecs.append(tv_i)
    out = []
    for rv, tv in zip(rvecs, tvecs):
        # refine from this candidate (stays on its branch)
        rv2, tv2 = cv2.solvePnPRefineLM(obj, corners, K, D, rv.copy(), tv.copy())
        R, _ = cv2.Rodrigues(rv2)
        proj, _ = cv2.projectPoints(obj, rv2, tv2, K, D)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - corners) ** 2, axis=1))))
        out.append((tv2.ravel(), R, err))
    return out


def choose_candidate(cands, T_base_cam, normal_prior_base=np.array([-1.0, 0.0, 0.0]),
                     max_prior_deg=60.0):
    """Drop solutions where the board does not face the robot (marker normal more
    than max_prior_deg from base -x), then take the lowest reprojection error.
    Returns (p, R, err_px, prior_angle_deg) or None."""
    ok = []
    for p, R, err in cands:
        z_base = T_base_cam[1] @ R[:, 2]
        ang = float(np.degrees(np.arccos(np.clip(np.dot(z_base, normal_prior_base), -1, 1))))
        if ang <= max_prior_deg:
            ok.append((p, R, err, ang))
    return min(ok, key=lambda c: c[2]) if ok else None


def board_in_base(T_base_flange, T_flange_cam, T_cam_marker, T_board_marker=None):
    """Board pose (cross centre, training convention) in the base frame."""
    if T_board_marker is None:
        T_board_marker = (P_BOARD_MARKER, R_BOARD_MARKER)
    T_base_marker = compose(compose(T_base_flange, T_flange_cam), T_cam_marker)
    return compose(T_base_marker, inv(*T_board_marker))


def average_poses(poses):
    """Mean position and the mean rotation matrix projected onto SO(3)."""
    P = np.mean([p for p, _ in poses], axis=0)
    U, _, Vt = np.linalg.svd(np.mean([R for _, R in poses], axis=0))
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return P, R


def rot_err_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def selftest():
    """Synthetic chain test: known pose -> projected corners -> PnP -> pose."""
    K = np.array([[772.55, 0, 320], [0, 772.55, 240], [0, 0, 1.0]])
    D = np.zeros(5)
    # board 0.84 m in front of the base, camera looking along base +x
    T_base_board = (np.array([0.84, 0.0, 0.02]), np.eye(3))
    T_base_flange = (np.array([0.40, 0.0, 0.04]), np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0.0]]))
    T_flange_cam = (P_FLANGE_CAM_WEBOTS, R_FLANGE_CAM_WEBOTS)
    T_base_marker = compose(T_base_board, (P_BOARD_MARKER, R_BOARD_MARKER))
    T_cam_marker = compose(inv(*compose(T_base_flange, T_flange_cam)), T_base_marker)
    # project the 4 corners
    obj = marker_obj()
    pc = (T_cam_marker[1] @ obj.T).T + T_cam_marker[0]
    assert np.all(pc[:, 2] > 0), "marker behind the camera"
    uv = (K @ (pc / pc[:, 2:3]).T).T[:, :2]
    T_base_cam = compose(T_base_flange, T_flange_cam)
    p, R, err, ang = choose_candidate(pnp_candidates(uv, K, D), T_base_cam)
    p_b, R_b = board_in_base(T_base_flange, T_flange_cam, (p, R))
    e_p = np.linalg.norm(p_b - T_base_board[0]); e_r = rot_err_deg(R_b, T_base_board[1])
    assert e_p < 1e-6 and e_r < 1e-4, (e_p, e_r)
    # 0.2 px corner noise, 200 trials, no branch flips
    rng = np.random.default_rng(0); worst = 0.0
    for _ in range(200):
        p, R, _, _ = choose_candidate(pnp_candidates(uv + rng.normal(0, 0.2, uv.shape), K, D), T_base_cam)
        worst = max(worst, rot_err_deg(board_in_base(T_base_flange, T_flange_cam, (p, R))[1], T_base_board[1]))
    assert worst < 5.0, worst
    print(f"aruco_board: 0.2 px noise -> worst board orientation error {worst:.2f} deg (no flips)")
    print(f"aruco_board: synthetic chain OK (|dp| {e_p:.1e} m, {e_r:.1e} deg); "
          f"marker {np.round(pc.mean(0), 3)} m from the camera, pixels {np.round(uv.mean(0), 1)}")


if __name__ == "__main__":
    selftest()

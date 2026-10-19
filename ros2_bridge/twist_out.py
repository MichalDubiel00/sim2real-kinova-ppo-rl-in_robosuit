#!/usr/bin/env python3
"""
Policy action -> twist (no ROS). Docs 6.56-6.57.

The policy outputs an OSC_POSE delta in [-1,1]^6 (base frame). In robosuite the
realised TCP velocity is a first-order filter (R^2 0.87-0.97 on arm_d):
    v[t+1] = alpha * v[t] + beta * a[t]
Defaults come from robosuite; on hardware the firmware adds its own lag, so retune
alpha/beta with the bridge's --step-test, or the lag counts twice.
Caps as in Gen3 twist mode: |v| <= 0.5 m/s, |w| <= 0.8727 rad/s, proportional.
"""

import numpy as np

# From hw_reference_arm_d_raw.npz (docs 6.56), order X, Y, Z, RX, RY, RZ.
ALPHA_ROBOSUITE = np.array([0.907, 0.926, 0.901, 0.744, 0.711, 0.734])
BETA_ROBOSUITE = np.array([0.0289, 0.0070, 0.0234, 0.1841, 0.1650, 0.1438])
V_CAP, W_CAP = 0.5, 0.8727


def cap_twist(v, w, v_cap=V_CAP, w_cap=W_CAP):
    """Proportional clip, translation and rotation separately (as the firmware)."""
    v, w = np.asarray(v, dtype=np.float64), np.asarray(w, dtype=np.float64)
    nv, nw = np.linalg.norm(v), np.linalg.norm(w)
    if nv > v_cap:
        v = v * (v_cap / nv)
    if nw > w_cap:
        w = w * (w_cap / nw)
    return v, w


class ActionToTwist:
    """Filter state + caps. step(a) -> (v[3], w[3]) in the BASE frame."""

    def __init__(self, alpha=ALPHA_ROBOSUITE, beta=BETA_ROBOSUITE,
                 v_cap=V_CAP, w_cap=W_CAP, speed_scale=1.0):
        self.alpha = np.asarray(alpha, dtype=np.float64)
        self.beta = np.asarray(beta, dtype=np.float64)
        self.v_cap, self.w_cap = v_cap, w_cap
        # speed_scale < 1 changes dynamics vs training: for checking directions only.
        self.speed_scale = float(speed_scale)
        self.reset()

    def reset(self):
        self.x = np.zeros(6)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64)[:6], -1.0, 1.0)
        self.x = self.alpha * self.x + self.beta * a
        v, w = cap_twist(self.x[:3], self.x[3:], self.v_cap, self.w_cap)
        # filter state stores the clipped value, or it would wind up above the cap
        self.x = np.concatenate([v, w])
        s = self.speed_scale
        return v * s, w * s


def base_to_tool(v_base, w_base, R_base_tool):
    """Twist from base frame to tool frame (ros2_kortex uses
    CARTESIAN_REFERENCE_FRAME_TOOL). R_base_tool: columns = tool axes in base."""
    Rt = np.asarray(R_base_tool, dtype=np.float64).T
    return Rt @ np.asarray(v_base, dtype=np.float64), Rt @ np.asarray(w_base, dtype=np.float64)


def shift_point(v, w, r):
    """Velocity of a point offset by r (same frame): v' = v + w x r.
    Used to move the policy's grip-point velocity to the robot's TCP."""
    return np.asarray(v) + np.cross(w, r), np.asarray(w)


def selftest():
    """Checks without ROS: python ros2_bridge/twist_out.py"""
    # 1. proportional caps keep direction
    v, w = cap_twist([3.0, 4.0, 0.0], [0.0, 0.0, 0.1])
    assert np.isclose(np.linalg.norm(v), V_CAP) and np.allclose(v / np.linalg.norm(v), [0.6, 0.8, 0])
    assert np.allclose(w, [0, 0, 0.1])
    # 2. constant action converges to min(beta/(1-alpha), cap)
    f = ActionToTwist()
    for _ in range(200):
        v, w = f.step([0, 0.3, 0, 0, 0, 0])
    assert np.isclose(v[1], 0.3 * BETA_ROBOSUITE[1] / (1 - ALPHA_ROBOSUITE[1]), rtol=1e-3), v
    for _ in range(200):
        v, w = f.step([1, 1, 1, 1, 1, 1])
    assert np.linalg.norm(v) <= V_CAP + 1e-9 and np.linalg.norm(w) <= W_CAP + 1e-9
    # 3. decays after zero action (does not hold)
    for _ in range(200):
        v, w = f.step(np.zeros(6))
    assert np.linalg.norm(v) < 1e-6 and np.linalg.norm(w) < 1e-6
    # 4. tool rotated 90 deg about base Z
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    vt, wt = base_to_tool([1, 0, 0], [0, 0, 1], Rz)
    assert np.allclose(vt, [0, -1, 0]) and np.allclose(wt, [0, 0, 1])
    # 5. point shift: rotation about Z, point 1 m on X -> velocity along +Y
    v2, _ = shift_point([0, 0, 0], [0, 0, 1.0], [1.0, 0, 0])
    assert np.allclose(v2, [0, 1, 0])
    print("twist_out: 5/5 tests OK")


if __name__ == "__main__":
    selftest()

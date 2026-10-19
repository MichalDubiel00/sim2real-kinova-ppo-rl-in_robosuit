#!/usr/bin/env python3
"""
Virtual wall and workspace box for the in-air test (no ROS). Docs 6.62.

Without the board nothing stops the arm: the policy was trained to push against
a button. Limits are in the training board frame (+x into the board, away from
the robot):
  wall  board x of the grip point <= -15 mm, just before the board's back plate
        (arm_d with a board never went deeper than -21 mm, p99.9 -26.5 mm)
  box   |y|, |z| <= 500 mm (measured max 263 / 318 mm); no limit on the Spot side
        by default - the body is physically there, as in training.

Braking: velocity toward a limit is capped at gap / t_brake, so the arm stops on
the limit. Overshoot > hard_margin means STOP.
Fails the robosuite criterion with a real board (3 attempts, docs 6.62):
use only in the air, --no-wall at the board.
"""

import numpy as np


class VirtualWall:
    def __init__(self, board_p, board_R, x_wall=-0.015, half_width=0.50, x_min=-np.inf,
                 hard_margin=0.02, t_brake=0.25):
        self.p = np.asarray(board_p, dtype=np.float64)
        self.R = np.asarray(board_R, dtype=np.float64)          # columns = board axes in base
        self.x_wall = float(x_wall)
        self.half = float(half_width)
        self.x_min = float(x_min)
        self.hard = float(hard_margin)
        self.t_brake = float(t_brake)
        self.clipped_steps = 0

    def to_board(self, p_base):
        return self.R.T @ (np.asarray(p_base, dtype=np.float64) - self.p)

    def violation(self, p_base):
        """(overshoot [m], name of the violated limit); 0 when inside."""
        x, y, z = self.to_board(p_base)
        worst, which = 0.0, ""
        for d, name in ((x - self.x_wall, "wall"), (self.x_min - x, "back"),
                        (abs(y) - self.half, "side"), (abs(z) - self.half, "top/bottom")):
            if d > worst:
                worst, which = d, name
        return worst, which

    def filter(self, p_base, v_base):
        """Returns (filtered v_base, stop, description). Rotation is not limited."""
        v_b = self.R.T @ np.asarray(v_base, dtype=np.float64)
        x, y, z = self.to_board(p_base)
        out = v_b.copy()
        clipped = False

        def brake(comp, gap):
            """Cap the component toward a limit (positive = toward it) at gap / t_brake."""
            nonlocal clipped
            vmax = max(gap, 0.0) / self.t_brake
            if comp > vmax:
                clipped = True
                return vmax
            return comp
        out[0] = brake(out[0], self.x_wall - x)
        out[0] = -brake(-out[0], x - self.x_min)
        for i, c in ((1, y), (2, z)):
            out[i] = brake(out[i], self.half - c)
            out[i] = -brake(-out[i], c + self.half)
        self.clipped_steps += int(clipped)
        over, which = self.violation(p_base)
        stop = over > self.hard
        desc = (f"STOP: {which} limit exceeded by {1000 * over:.0f} mm" if stop else
                ("clipped" if clipped else ""))
        return self.R @ out, stop, desc


def wall_for(core, board_p, board_R, **kw):
    """`core` kept for signature compatibility."""
    return VirtualWall(board_p, board_R, **kw)


def selftest():
    R = np.eye(3)
    w = VirtualWall(np.array([0.84, 0.0, 0.02]), R)
    assert abs(w.x_wall - (-0.015)) < 1e-9
    # 1. far from the wall: untouched
    v, stop, _ = w.filter([0.55, 0.0, 0.02], [0.3, 0.1, -0.1])
    assert np.allclose(v, [0.3, 0.1, -0.1]) and not stop
    # 2. just before the wall, moving in: x braked, rest kept
    p = [0.84 - 0.02, 0.0, 0.02]
    v, stop, d = w.filter(p, [0.3, 0.1, 0.0])
    assert abs(v[0] - 0.005 / 0.25) < 1e-9 and v[1] == 0.1 and not stop and d == "clipped"
    # 3. same pose, moving away: allowed
    v, _, _ = w.filter(p, [-0.3, 0.0, 0.0])
    assert v[0] == -0.3
    # 4. 3 cm through the wall: STOP
    v, stop, d = w.filter([0.84 - 0.015 + 0.03, 0.0, 0.02], [0.0, 0.0, 0.0])
    assert stop and "wall" in d
    # 5. side limit: outward blocked, inward allowed
    v, _, _ = w.filter([0.70, 0.51, 0.02], [0.0, 0.2, 0.0])
    assert v[1] == 0.0
    v, _, _ = w.filter([0.70, 0.51, 0.02], [0.0, -0.2, 0.0])
    assert v[1] == -0.2
    # 6. board yawed 10 deg: wall evaluated in the board frame
    a = np.radians(10); Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    w2 = VirtualWall(np.array([0.84, 0.0, 0.02]), Rz)
    p_in = np.array([0.84, 0.0, 0.02]) + Rz @ np.array([-0.02, 0.0, 0.0])
    v, _, _ = w2.filter(p_in, Rz @ np.array([0.3, 0.0, 0.0]))
    assert np.allclose(v, Rz @ np.array([0.005 / 0.25, 0.0, 0.0]), atol=1e-12)
    # 7. lagged arm (L 40 ms, tau 60 ms) at 0.5 m/s into the wall: overshoot < 10 mm
    #    (~5 cm without braking). On hardware pick t_brake ~ 2.5 * (L + tau) from the step test.
    dt, pos, vel, buf = 0.002, np.array([0.84 - 0.20, 0.0, 0.02]), np.zeros(3), []
    worst = -1.0
    for k in range(int(3.0 / dt)):
        if k % 25 == 0:                                       # 20 Hz bridge loop
            cmd, _, _ = w.filter(pos, np.array([0.5, 0.0, 0.0]))
        buf.append(cmd.copy())
        delayed = buf[-20] if len(buf) >= 20 else np.zeros(3)   # 40 ms dead time
        vel += dt / 0.06 * (delayed - vel)
        pos = pos + vel * dt
        worst = max(worst, w.to_board(pos)[0] - w.x_wall)
    assert worst < 0.010, worst
    print(f"safety: 7/7 tests OK (lagged arm: max {1000 * worst:+.1f} mm relative to the wall)")


if __name__ == "__main__":
    selftest()

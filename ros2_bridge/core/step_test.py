#!/usr/bin/env python3
"""
Twist step test: identify the arm and retune the bridge filter (no ROS). Docs 6.62.

ActionToTwist reproduces robosuite dynamics; the firmware adds its own, so we
measure the arm and pick a filter such that (bridge filter -> arm) ~ robosuite.

Per axis, first order plus dead time: v(t) = K * u(t - L) through 1/(tau*s + 1).
  L    dead time - not removable, reported against the policy budget (6.52)
  tau  subtracted from the robosuite time constant
  K    divided out of the filter gain
robosuite: tau_rs = -dt / ln(alpha_rs), K_rs = beta_rs / (1 - alpha_rs)
bridge:    tau_br = max(tau_rs - tau, 0), alpha_br = exp(-dt / tau_br),
           beta_br = K_rs * (1 - alpha_br) / K
Pulses per axis: +A, pause, -A, pause (0.05 m/s / 0.2 rad/s, 0.8 s).
"""

import json

import numpy as np

AXES = ["vx", "vy", "vz", "wx", "wy", "wz"]


def pulse_plan(amp_lin=0.05, amp_ang=0.2, t_on=0.8, t_off=1.0, axes=range(6)):
    """List of (axis, value, duration) segments; axis -1 = pause."""
    plan = [(-1, 0.0, t_off)]
    for ax in axes:
        A = amp_lin if ax < 3 else amp_ang
        plan += [(ax, +A, t_on), (-1, 0.0, t_off), (ax, -A, t_on), (-1, 0.0, t_off)]
    return plan


def plan_to_cmd(plan):
    """Plan -> u(t) returning a 6-vector, plus [(t_start, t_end, axis, A)]."""
    segs, t = [], 0.0
    for ax, val, dur in plan:
        segs.append((t, t + dur, ax, val))
        t += dur
    def u(tt):
        out = np.zeros(6)
        for t0, t1, ax, val in segs:
            if t0 <= tt < t1 and ax >= 0:
                out[ax] = val
        return out
    return u, segs, t


def velocity_from_poses(t, P, Rm):
    """Linear [m/s] and angular [rad/s] base-frame velocity from grip-point
    poses by central difference. t[N], P[N,3], Rm[N,3,3]."""
    t = np.asarray(t, dtype=np.float64)
    V = np.zeros((len(t), 3)); W = np.zeros((len(t), 3))
    for i in range(1, len(t) - 1):
        dt = t[i + 1] - t[i - 1]
        if dt <= 0:
            continue
        V[i] = (P[i + 1] - P[i - 1]) / dt
        dR = Rm[i + 1] @ Rm[i - 1].T
        ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        if ang > 1e-9:
            ax = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * np.sin(ang))
            W[i] = ax * ang / dt
    return np.hstack([V, W])


def analyze(t, Y, segs):
    """Per pulse: dead time L, time constant tau, gain K (mean of the last 30 %
    / command). Returns axis -> medians over that axis's pulses."""
    t = np.asarray(t, dtype=np.float64)
    res = {}
    for t0, t1, ax, A in segs:
        if ax < 0:
            continue
        m = (t >= t0) & (t <= t1)
        if m.sum() < 5:
            continue
        tt, y = t[m] - t0, Y[m, ax] / A                          # normalised response
        tail = y[tt >= 0.7 * (t1 - t0)]
        K = float(np.median(tail)) if len(tail) else float("nan")
        if not np.isfinite(K) or K <= 0.05:
            res.setdefault(ax, []).append((np.nan, np.nan, K))
            continue
        yn = y / K
        i10 = np.argmax(yn >= 0.10); i63 = np.argmax(yn >= 0.632)
        # FOPDT: t10 = L + tau*ln(1/0.9), t63 = L + tau. A plain 10 % threshold
        # overestimates L by ~0.1*tau.
        if yn.max() >= 0.632:
            t10, t63 = float(tt[i10]), float(tt[i63])
            tau = (t63 - t10) / (1.0 - np.log(1 / 0.9))
            L = t63 - tau
        else:
            L = tau = np.nan
        res.setdefault(ax, []).append((L, max(tau, 0.0) if np.isfinite(tau) else np.nan, K))
    out = {}
    for ax, rows in res.items():
        r = np.array(rows, dtype=np.float64)
        out[ax] = dict(L=float(np.nanmedian(r[:, 0])), tau=float(np.nanmedian(r[:, 1])),
                       K=float(np.nanmedian(r[:, 2])), n=len(rows))
    return out


def fit_filter(meas, alpha_rs, beta_rs, dt=0.05):
    """New bridge filter alpha/beta from the arm measurement (see header)."""
    alpha_br, beta_br, notes = np.array(alpha_rs, float), np.array(beta_rs, float), []
    for ax in range(6):
        if ax not in meas or not np.isfinite(meas[ax]["tau"]) or not np.isfinite(meas[ax]["K"]):
            notes.append(f"{AXES[ax]}: no measurement, robosuite filter")
            continue
        a, b = alpha_rs[ax], beta_rs[ax]
        tau_rs = -dt / np.log(a)
        K_rs = b / (1 - a)
        tau_br = max(tau_rs - meas[ax]["tau"], 0.0)
        a_br = float(np.exp(-dt / tau_br)) if tau_br > 1e-6 else 0.0
        b_br = K_rs * (1 - a_br) / meas[ax]["K"]
        alpha_br[ax], beta_br[ax] = a_br, b_br
        notes.append(f"{AXES[ax]}: tau_rs {tau_rs:.3f} s, arm tau {meas[ax]['tau']:.3f} s L {1000 * meas[ax]['L']:.0f} ms "
                     f"K {meas[ax]['K']:.2f} -> alpha {a:.3f}->{a_br:.3f}, beta {b:.4f}->{b_br:.4f}"
                     + (" | NOTE: arm slower than robosuite, the filter cannot compensate" if meas[ax]["tau"] > tau_rs else ""))
    return alpha_br, beta_br, notes


def save_filter(path, alpha, beta, meas, notes):
    json.dump(dict(alpha=list(map(float, alpha)), beta=list(map(float, beta)),
                   pomiar={AXES[k]: v for k, v in meas.items()}, uwagi=notes),
              open(path, "w"), indent=1, ensure_ascii=False)


def load_filter(path):
    d = json.load(open(path))
    return np.array(d["alpha"]), np.array(d["beta"])


def selftest():
    """Synthetic arm: L = 60 ms, tau = 80 ms, K = 0.9 on all axes."""
    from twist_out import ALPHA_ROBOSUITE, BETA_ROBOSUITE
    L, tau, K, h = 0.06, 0.08, 0.9, 0.002
    u, segs, T = plan_to_cmd(pulse_plan())
    t = np.arange(0, T, h)
    Y = np.zeros((len(t), 6)); y = np.zeros(6)
    for i, tt in enumerate(t):
        y = y + h / tau * (K * u(tt - L) - y)
        Y[i] = y
    t_s, Y_s = t[::8], Y[::8]                                 # sampled like /joint_states (~60 Hz)
    meas = analyze(t_s, Y_s, segs)
    for ax in range(6):
        m = meas[ax]
        assert abs(m["K"] - K) < 0.02, m
        assert abs(m["L"] - L) < 0.03 and abs(m["tau"] - tau) < 0.04, (AXES[ax], m)
    a, b, notes = fit_filter(meas, ALPHA_ROBOSUITE, BETA_ROBOSUITE)
    # static chain gain (filter * arm) = robosuite
    for ax in range(6):
        k_chain = b[ax] / (1 - a[ax]) * meas[ax]["K"]
        k_rs = BETA_ROBOSUITE[ax] / (1 - ALPHA_ROBOSUITE[ax])
        assert abs(k_chain - k_rs) / k_rs < 1e-6
    # chain time constant ~ robosuite
    tau_rs = -0.05 / np.log(ALPHA_ROBOSUITE[0]); tau_br = -0.05 / np.log(a[0])
    assert abs((tau_br + meas[0]["tau"]) - tau_rs) < 1e-6
    # measured from pose-derived velocity
    P = np.cumsum(Y_s[:, :3], axis=0) * (8 * h)
    V = velocity_from_poses(t_s, P, np.repeat(np.eye(3)[None], len(t_s), 0))
    assert np.max(np.abs(V[2:-2, 0] - Y_s[2:-2, 0])) < 0.01
    print(f"step_test: 4/4 tests OK (measured L {1000 * meas[0]['L']:.0f} ms, tau {1000 * meas[0]['tau']:.0f} ms, "
          f"K {meas[0]['K']:.2f}; vx filter alpha {ALPHA_ROBOSUITE[0]:.3f} -> {a[0]:.3f})")


if __name__ == "__main__":
    selftest()

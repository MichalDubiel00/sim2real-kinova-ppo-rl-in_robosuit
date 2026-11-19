"""
compare_models.py plus arm jitter, measured at physics rate (docs 6.72).

Takes the same arguments as compare_models.py, plus --jitter-json FILE. Metrics per episode,
over the motion until the episode ends:
  tremor_mm   RMS of the TCP position high-passed at HP_HZ [mm]: what the eye reads as shaking
  ripple      RMS of TCP speed high-passed at HP_HZ / mean speed: pulsing inside the motion
  rev_tcp     TCP direction reversals per second (control rate, both speeds above V_MIN)
  rev_joint   joint velocity sign flips per second, summed over joints (above W_MIN)
  act_flip    share of control steps where an action component flips sign (both above A_MIN)
"""

import json
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import compare_models as cm  # noqa: E402  (sets the TF guard before SB3)
from envs.button_board import ButtonBoardEnv  # noqa: E402

HP_HZ = 3.0
V_MIN = 0.005   # m/s
W_MIN = 0.01    # rad/s
A_MIN = 0.05


def _pop_arg(name):
    if name not in sys.argv:
        raise SystemExit(f"missing {name}")
    i = sys.argv.index(name)
    val = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return val


def metrics(pos, vel, qd, act, fs, hz):
    pos, vel, qd, act = (np.asarray(x, dtype=np.float64) for x in (pos, vel, qd, act))
    dur = len(pos) / fs
    sos = butter(4, HP_HZ, "highpass", fs=fs, output="sos")
    tremor = sosfiltfilt(sos, pos, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    sub = max(1, round(fs / hz))
    v, w = vel[::sub], qd[::sub]
    both = (np.linalg.norm(v[1:], axis=1) > V_MIN) & (np.linalg.norm(v[:-1], axis=1) > V_MIN)
    rev = both & (np.sum(v[1:] * v[:-1], axis=1) < 0)
    flips = (w[1:] * w[:-1] < 0) & (np.abs(w[1:]) > W_MIN) & (np.abs(w[:-1]) > W_MIN)
    a = act[:, :6]
    aflip = (a[1:] * a[:-1] < 0) & (np.abs(a[1:]) > A_MIN) & (np.abs(a[:-1]) > A_MIN)
    return {
        "tremor_mm": 1000 * float(np.sqrt(np.mean(np.sum(tremor ** 2, axis=1)))),
        "ripple": float(np.sqrt(np.mean(sosfiltfilt(sos, speed) ** 2)) / max(speed.mean(), 1e-6)),
        "mean_speed": float(speed.mean()),
        "rev_tcp": float(rev.sum() / dur),
        "rev_joint": float(flips.sum() / dur),
        "act_flip": float(aflip.any(axis=1).mean()),
    }


def main():
    out = _pop_arg("--jitter-json")
    rows, log = [], {}
    upd, step = ButtonBoardEnv._update_observables, ButtonBoardEnv.step

    def update(self, force=False):
        upd(self, force)
        if not force and log:
            r = self.robots[0]
            site = self.sim.model.site_id2name(r.eef_site_id[r.arms[0]])
            log["pos"].append(self.sim.data.get_site_xpos(site).copy())
            log["vel"].append(self.sim.data.get_site_xvelp(site).copy())
            log["qd"].append(self.sim.data.qvel[r._ref_joint_vel_indexes].copy())

    def env_step(self, action):
        if log:
            log["act"].append(np.array(action, dtype=np.float64))
        return step(self, action)

    ButtonBoardEnv._update_observables = update
    ButtonBoardEnv.step = env_step
    orig_run = cm.run_episode

    def run_episode(env, model, seed, action_ema=None):
        log.clear()
        log.update(pos=[], vel=[], qd=[], act=[])
        r = orig_run(env, model, seed, action_ema)
        m = metrics(log["pos"], log["vel"], log["qd"], log["act"],
                    1.0 / env.model_timestep, env.control_freq)
        m.update(seed=seed, success=r["success"], steps=r["steps"], action_rate=r["action_rate"])
        rows.append(m)
        return r

    cm.run_episode = run_episode
    try:
        cm.main()
    finally:
        with open(out, "w") as f:
            json.dump(rows, f)
        if rows:
            print("jitter (median [p90] over episodes):")
            for k in ("tremor_mm", "ripple", "mean_speed", "rev_tcp", "rev_joint", "act_flip",
                      "action_rate"):
                v = [r[k] for r in rows]
                print(f"  {k:12s} {np.median(v):8.3f}  [{np.percentile(v, 90):8.3f}]")


if __name__ == "__main__":
    main()

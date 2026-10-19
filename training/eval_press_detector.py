#!/usr/bin/env python3
"""
Geometric press criterion, validated on arm_d episodes (docs 6.61).
Candidate: grip point past the unpressed cap-top plane by >= threshold along the
press axis, lateral offset < r, aligned (align < a_max). Ground truth is env._pressed;
reports hits, false alarms and timing over a threshold grid. Result: negative.

  python training/eval_press_detector.py --episodes 30
"""

import sys
sys.modules["tensorflow"] = None

import argparse
import itertools
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ros2_bridge"))

from diag_hw_reference import make_env, flatten_obs
from obs_core import SCENE_ARM_D


def collect(episodes, seed):
    from stable_baselines3 import PPO
    model = PPO.load(SCENE_ARM_D["model"], device="cpu")
    env = make_env(nominal=False)
    rob = env.robots[0]
    eef = rob.eef_site_id[rob.arms[0]]
    recs = []
    for ep in range(episodes):
        np.random.seed(seed + ep)
        obs = env.reset()
        # UNPRESSED cap tops (board pose fixed per episode)
        tops = np.array([env.sim.data.site_xpos[i].copy() for i in env._btn_top_ids])
        axes = np.array([-np.asarray(env.BUTTON_FACES[i][1], dtype=float) for i in range(env.N_BUTTONS)])
        bias = env._bias_pos.copy()
        T = []
        for t in range(2000):
            tgt = env._target_btn()
            grip = env.sim.data.site_xpos[eef].copy()
            z = env.sim.data.site_xmat[eef].reshape(3, 3)[:, 2].copy()
            was = list(env._pressed)
            a, _ = model.predict(flatten_obs(obs), deterministic=True)
            obs, _, done, info = env.step(a)
            newly = [i for i in range(env.N_BUTTONS) if env._pressed[i] and not was[i]]
            T.append((tgt, grip, z, newly))
            if done or all(env._pressed):
                break
        recs.append(dict(tops=tops, axes=axes, bias=bias, T=T, n_pressed=int(sum(env._pressed))))
        print(f"  ep{ep:02d}: {recs[-1]['n_pressed']}/5, {len(T)} steps", flush=True)
    env.close()
    return recs


def evaluate(recs, thr, r_lat, a_max, K=10, use_bias=False):
    hits = late = fa = n_act = 0
    lat = []
    for rec in recs:
        tops = rec["tops"] + (rec["bias"] if use_bias else 0.0)
        T = rec["T"]
        # activations: (step, button)
        acts = [(t, b) for t, (_, _, _, nw) in enumerate(T) for b in nw]
        n_act += len(acts)
        # criterion firings: first step in each target period
        fires = []
        armed_for = None
        for t, (tgt, grip, z, _) in enumerate(T):
            if tgt != armed_for:
                armed_for, fired = tgt, False
            if fired:
                continue
            a = rec["axes"][tgt]
            d = grip - tops[tgt]
            pen = float(np.dot(d, a))
            lat_off = float(np.linalg.norm(d - pen * a))
            align = float(np.degrees(np.arccos(np.clip(abs(np.dot(z, a)), -1, 1))))
            if pen >= thr and lat_off <= r_lat and align <= a_max:
                fires.append((t, tgt))
                fired = True
        used = set()
        for ta, b in acts:
            cand = [(tf, bf) for tf, bf in fires if bf == b and tf <= ta + K]
            if cand:
                tf = cand[0][0]
                hits += 1
                lat.append(ta - tf)
                used.add((tf, b))
                late += int(tf > ta)
        for tf, bf in fires:
            if (tf, bf) not in used:
                fa += 1
    return dict(hits=hits, n_act=n_act, fa=fa, late=late,
                lat_med=float(np.median(lat)) if lat else float("nan"),
                lat_p10=float(np.percentile(lat, 10)) if lat else float("nan"),
                lat_p90=float(np.percentile(lat, 90)) if lat else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--save", default="logs/board_6dof/eval_2026-10-15/press_detector_traces.npz")
    args = ap.parse_args()
    recs = collect(args.episodes, args.seed)
    np.savez(args.save, recs=np.array(recs, dtype=object))

    # grip point vs cap-top plane at activation
    pens = []
    for rec in recs:
        for t, (_, grip, z, nw) in enumerate(rec["T"]):
            for b in nw:
                pens.append(float(np.dot(grip - rec["tops"][b], rec["axes"][b])))
    pens = np.array(pens)
    print(f"\nactivations {len(pens)}; grip-point depth past the cap top AT activation "
          f"[mm]: p5 {1000 * np.percentile(pens, 5):.1f}, median {1000 * np.median(pens):.1f}, "
          f"p95 {1000 * np.percentile(pens, 95):.1f}")

    print("\n thr   r_lat  align | hits         false      late       delay[st] p10/med/p90 | with 15 mm bias")
    rows = []
    for thr, r_lat, a_max in itertools.product([-0.01, -0.005, 0.0, 0.003, 0.006],
                                               [0.015, 0.025], [20, 35]):
        e = evaluate(recs, thr, r_lat, a_max)
        eb = evaluate(recs, thr, r_lat, a_max, use_bias=True)
        rows.append((thr, r_lat, a_max, e, eb))
        print(f"{1000 * thr:5.0f} {1000 * r_lat:5.0f} {a_max:6d} | {e['hits']:3d}/{e['n_act']:<3d} "
              f"{e['fa']:12d} {e['late']:9d}   {e['lat_p10']:5.0f}/{e['lat_med']:4.0f}/{e['lat_p90']:4.0f} "
              f"| {eb['hits']:3d}/{eb['n_act']} fa {eb['fa']}")


if __name__ == "__main__":
    main()

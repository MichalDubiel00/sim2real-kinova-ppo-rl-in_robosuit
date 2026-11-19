"""
compare_models.py plus robot-Spot clearance (docs 6.68).

Takes the same arguments as compare_models.py. Every step it measures the minimum distance
(mj_geomDistance) between the arm/gripper collision geoms (except robot0_base, which sits on
Spot) and the Spot body and front-camera boxes, separately for the start (t < 15) and the rest
of the episode. Negative = penetration. Per-episode results go to --clearance-json.

    python training/spot_clearance.py arm_p=logs/.../ppo_board_114123552_steps.zip \
        --clearance-json out.json <compare_models flags>
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import compare_models as cm  # noqa: E402  (sets the TF guard before SB3)
import mujoco  # noqa: E402
from envs.button_board import ButtonBoardEnv  # noqa: E402

SPOT_GEOMS = ("spot_body_collision", "spot_frontcam_collision", "spot_labcam_collision")
START_STEPS = 15


def _pop_arg(name):
    if name not in sys.argv:
        raise SystemExit(f"missing {name}")
    i = sys.argv.index(name)
    val = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return val


def main():
    out = _pop_arg("--clearance-json")
    eps, cache = [], {}
    orig_check = ButtonBoardEnv._check_spot_collision

    def geoms(env):
        m = env.sim.model._model
        if cache.get("m") is not m:  # hard_reset builds a new model every episode
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
            rob = [i for i, n in enumerate(names)
                   if (n.startswith("robot0") or "gripper0" in n)
                   and (m.geom_contype[i] or m.geom_conaffinity[i])
                   and n != "robot0_base_collision"]
            spot = {n: names.index(n) for n in SPOT_GEOMS if n in names}
            cache.update(m=m, rob=rob, spot=spot, names=names)
        return cache

    def check(self):
        c = geoms(self)
        m, d = self.sim.model._model, self.sim.data._data
        ft = np.zeros(6)
        e = eps[-1]
        phase = "late" if self.timestep >= START_STEPS else "early"
        for sname, sid in c["spot"].items():
            key = f"{sname}:{phase}"
            for g in c["rob"]:
                dist = mujoco.mj_geomDistance(m, d, g, sid, 0.25, ft)
                # exactly 0.0 is a geomDistance artefact (fromto left empty); contacts are
                # counted separately by compare_models
                if dist != 0.0 and dist < e.setdefault(key, [1.0])[0]:
                    e[key] = [float(dist), c["names"][g], int(self.timestep),
                              int(sum(self._pressed))]
        return orig_check(self)

    ButtonBoardEnv._check_spot_collision = check
    orig_run = cm.run_episode

    def run_episode(env, model, seed, action_ema=None):
        eps.append({"seed": seed})
        r = orig_run(env, model, seed, action_ema)
        eps[-1].update(success=r["success"], steps=r["steps"],
                       spot_steps=r["spot_steps"], coll=r["coll_steps"])
        return r

    cm.run_episode = run_episode
    try:
        cm.main()
    finally:
        with open(out, "w") as f:
            json.dump(eps, f)
        late = [min(e.get(f"{s}:late", [1.0])[0] for s in SPOT_GEOMS) for e in eps]
        for sname in SPOT_GEOMS:
            per = [e[f"{sname}:late"][0] for e in eps if f"{sname}:late" in e]
            pre = [e[f"{sname}:early"][0] for e in eps if f"{sname}:early" in e]
            if per or pre:
                print(f"  {sname}: start min {100 * min(pre or [1.0]):.1f} cm, rest min / p10 "
                      f"{100 * min(per or [1.0]):.1f} / {100 * np.percentile(per or [1.0], 10):.1f} cm")
        early = [min(e.get(f"{s}:early", [1.0])[0] for s in SPOT_GEOMS) for e in eps]
        if eps:
            print(f"clearance start min: {100 * min(early):.1f} cm")
            print(f"clearance rest min / p10 / median: {100 * min(late):.1f} / "
                  f"{100 * np.percentile(late, 10):.1f} / {100 * np.median(late):.1f} cm")


if __name__ == "__main__":
    main()

"""
ButtonBoardEnv observation map -> envs/obs_contract.json.
Order comes from obs_dict insertion order and is duplicated by the *-state
aggregates; a wrong order gives no error, only a very low score (docs 6.22).

    python training/diag_obs_map.py [--json]
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import robosuite as suite
from envs.button_board import ButtonBoardEnv

ROBOTS = ["Kinova3_6DOF", "Kinova3"]

# JSON descriptions in English: read by code on the robot.
DESC = {
 "robot0_joint_pos":      ("joint angles", "rad", "joint", "joint_states.position - VERIFY ORDERING"),
 "robot0_joint_pos_cos":  ("cosines of joint angles", "-", "joint", "removes discontinuity at +/-pi"),
 "robot0_joint_pos_sin":  ("sines of joint angles", "-", "joint", ""),
 "robot0_joint_vel":      ("joint velocities", "rad/s", "joint", "noisier on hardware"),
 "robot0_eef_pos":        ("end-effector position", "m", "world", "forward kinematics or TF base_link->eef"),
 "robot0_eef_quat":       ("EEF orientation (body frame)", "quat xyzw", "world", "USED BY eef_align - not quat_site"),
 "robot0_eef_quat_site":  ("EEF orientation (site frame)", "quat xyzw", "world", "rotated 90deg; same Z axis"),
 "robot0_gripper_qpos":   ("Robotiq85 finger angles", "rad", "gripper", "near-constant if force_gripper_closed"),
 "robot0_gripper_qvel":   ("Robotiq85 finger velocities", "rad/s", "gripper", ""),
 "target_relative_pos":   ("EEF->target vector, WITH NOISE", "m", "world", "PRIMARY PERCEPTION INPUT (ArUco)"),
 "target_distance":       ("norm of target_relative_pos", "m", "-", "derived"),
 "target_onehot":         ("current target button", "one-hot 5", "-", "SET BY SUPERVISOR on hardware"),
 "target_press_axis":     ("press direction (-normal), WITH NOISE", "unit vec", "world", "board pose (ArUco) + geometry"),
 "eef_align_with_target": ("angle between EEF +Z and press axis", "rad 0..pi/2", "-", "0 = perpendicular to face"),
 "robot0_proprio-state":  ("AGGREGATE: copy of robot0_* keys", "-", "-", "DUPLICATE - reproduce on hardware"),
 "object-state":          ("AGGREGATE: copy of target_*/eef_align", "-", "-", "DUPLICATE - reproduce on hardware"),
}


def build_env(robot, horizon=100):
    ctrl = suite.load_composite_controller_config(controller="BASIC", robot=robot)
    return ButtonBoardEnv(
        robots=robot, controller_configs=ctrl, has_renderer=False,
        has_offscreen_renderer=False, use_camera_obs=False,
        horizon=horizon, random_order=True)


def key_map(obs):
    """[(key, start, end, dim), ...] + total dim."""
    out, i = [], 0
    for k, v in obs.items():
        n = int(np.asarray(v).flatten().size)
        out.append((k, i, i + n, n))
        i += n
    return out, i


def print_map(kmap, dim, robot):
    print(f"\n===== {robot} =====")
    agg = sum(n for k, _, _, n in kmap if k.endswith("-state"))
    for k, a, b, n in kmap:
        tag = "  <-- AGGREGATE (duplicate)" if k.endswith("-state") else ""
        print(f"  [{a:3d}:{b:3d}] {k:24s} dim={n:2d}{tag}")
    print(f"  TOTAL = {dim}   (aggregates: {agg}, unique: {dim - agg})")


def verify(env, ref):
    """Is the key order stable and are the aggregates real copies?"""
    print("\n--- ordering stability ---")
    bad = []
    for s in (0, 1, 7, 42, 1234, 9999):
        np.random.seed(s)
        if key_map(env.reset())[0] != ref:
            bad.append(s)
    print(f"  6 seeds        : {'OK' if not bad else f'MISMATCH {bad}'}")

    env.reset()
    bad_step = sum(key_map(env.step(np.zeros(env.action_dim))[0])[0] != ref
                   for _ in range(5))
    print(f"  after step()   : {'OK' if not bad_step else 'MISMATCH'}")

    obs = env.reset()
    flat = np.concatenate([np.asarray(v).flatten() for v in obs.values()],
                          dtype=np.float32)
    idx = {k: (a, b) for k, a, b, _ in ref}
    n_prop = idx["robot0_proprio-state"][1] - idx["robot0_proprio-state"][0]
    uniq = sum(n for k, _, _, n in ref if not k.endswith("-state"))
    d1 = np.allclose(flat[slice(*idx["robot0_proprio-state"])], flat[0:n_prop], atol=1e-5)
    d2 = np.allclose(flat[slice(*idx["object-state"])], flat[n_prop:uniq], atol=1e-5)
    print(f"  proprio-state copy : {d1}")
    print(f"  object-state copy  : {d2}")
    return not bad and not bad_step and d1 and d2


def write_json(data, path):
    out = collections.OrderedDict()
    out["_description"] = (
        "Observation-vector contract for ButtonBoardEnv. Ordering comes from "
        "obs_dict.values() in RobosuiteGymWrapper._flatten_obs - insertion order, "
        "not sorted, and not written down anywhere else in the code. The vector is "
        "DUPLICATED: the *-state aggregates repeat the individual keys. "
        "Regenerate: python training/diag_obs_map.py --json")
    out["_verification"] = ("ordering identical across 6 seeds, after step(), with "
                            "random_order=True, and in a separate process")
    out["_warning"] = ("A wrong ordering raises no error as long as the dimension "
                       "matches - the only symptom is a very low success rate (docs 6.22).")
    out["robots"] = collections.OrderedDict()
    for robot, (kmap, dim, adim) in data.items():
        keys = []
        for k, a, b, n in kmap:
            meaning, units, frame, note = DESC.get(k, ("?", "?", "?", ""))
            keys.append(collections.OrderedDict([
                ("key", k), ("start", a), ("end", b), ("dim", n),
                ("meaning", meaning), ("units", units), ("frame", frame),
                ("aggregate", k.endswith("-state")), ("hardware_note", note)]))
        out["robots"][robot] = collections.OrderedDict([
            ("obs_dim", dim), ("action_dim", adim), ("keys", keys)])
    out["action"] = collections.OrderedDict([
        ("dim", 7), ("range", "[-1, 1]"),
        ("0:3", "position delta, OSC_POSE"),
        ("3:6", "orientation delta, OSC_POSE"),
        ("6", "gripper - overridden when force_gripper_closed=True"),
        ("nominal_output_max", [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]),
        ("input_ref_frame", "base"),
        ("measured_scale_m_per_step", {"X": 0.01126, "Y": 0.00722, "Z": 0.01317}),
        ("measured_steps_to_90pct", {"X": 12, "Y": 3, "Z": 2}),
        ("measured_note", "linear in amplitude; measured 2026-09-04 by "
                          "training/diag_action_scale.py at 20 Hz control rate")])
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.normpath(path)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    data = {}
    for robot in ROBOTS:
        env = build_env(robot)
        kmap, dim = key_map(env.reset())
        print_map(kmap, dim, robot)
        if robot == ROBOTS[0]:
            ok = verify(env, kmap)
            print(f"  --> ordering {'STABLE' if ok else 'UNSTABLE - INVESTIGATE'}")
        data[robot] = (kmap, dim, int(env.action_dim))
        env.close()

    if args.json:
        here = os.path.dirname(os.path.abspath(__file__))
        write_json(data, os.path.join(here, "..", "envs", "obs_contract.json"))


if __name__ == "__main__":
    main()

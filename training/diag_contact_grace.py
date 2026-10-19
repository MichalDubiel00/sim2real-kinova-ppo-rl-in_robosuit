"""
Contact quality diagnostics (v2.4 vs v2.3).
    python training/diag_contact_grace.py MODEL.zip --episodes 20 --seed 777

Per finger-button contact step:
  bumping       contact with a non-target, not yet pressed button
  lingering     contact with an already pressed non-target (post-press retreat);
                reported separately - counting it as bumping inflated the metric ~13x
  contact_speed EEF speed at contact [m/s]
  joint_asymmetry |left - right inner finger joint| [rad] at contact
"""
import sys
sys.modules["tensorflow"] = None

import argparse
import numpy as np
from stable_baselines3 import PPO

from training.train_board import RobosuiteGymWrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=str)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--horizon", type=int, default=500)
    args = p.parse_args()

    gym_env = RobosuiteGymWrapper(dict(
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, horizon=args.horizon,
    ))
    env = gym_env._env
    model = PPO.load(args.model)

    lif_id = env.sim.model.joint_name2id("gripper0_right_left_inner_finger_joint")
    rif_id = env.sim.model.joint_name2id("gripper0_right_right_inner_finger_joint")
    lif_adr = env.sim.model.jnt_qposadr[lif_id]
    rif_adr = env.sim.model.jnt_qposadr[rif_id]

    contact_steps_total = 0
    bump_steps = 0       # contact with an unpressed non-target (real bumping)
    linger_steps = 0     # contact only with already pressed non-targets (retreat)
    speeds = []
    asymmetries = []
    ep_success = []
    ep_bump_frac = []

    for ep in range(args.episodes):
        np.random.seed(args.seed + ep)
        obs, _ = gym_env.reset()
        done = False
        ep_contact = 0
        ep_bump = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = gym_env.step(action)
            done = terminated or truncated

            touched = env._finger_touched_buttons()
            if touched:
                contact_steps_total += 1
                ep_contact += 1
                tgt = env._target_btn()
                wrong = [i for i in touched if i != tgt]
                if wrong:
                    if any(not env._pressed[i] for i in wrong):
                        bump_steps += 1
                        ep_bump += 1
                    else:
                        linger_steps += 1
                speed = float(np.linalg.norm(
                    env.sim.data.get_site_xvelp(env._eef_site_name)))
                speeds.append(speed)
                asym = abs(float(env.sim.data.qpos[lif_adr])
                           - float(env.sim.data.qpos[rif_adr]))
                asymmetries.append(asym)

        ep_success.append(all(env._pressed))
        ep_bump_frac.append(ep_bump / ep_contact if ep_contact else 0.0)

    speeds = np.array(speeds) if speeds else np.array([0.0])
    asymmetries = np.array(asymmetries) if asymmetries else np.array([0.0])

    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}, successes: {sum(ep_success)}/{args.episodes}")
    print(f"Finger-button contact steps: {contact_steps_total}")
    print(f"Bumping an UNPRESSED non-target: {bump_steps} "
          f"({100.0 * bump_steps / max(1, contact_steps_total):.1f}%)")
    print(f"Lingering contact with a pressed non-target (post-success retreat): "
          f"{linger_steps} "
          f"({100.0 * linger_steps / max(1, contact_steps_total):.1f}%)")
    print(f"Mean bumping fraction per episode: "
          f"{np.mean(ep_bump_frac):.3f}")
    print(f"EEF speed at contact [m/s]: mean={speeds.mean():.3f} "
          f"median={np.median(speeds):.3f} p95={np.percentile(speeds, 95):.3f} "
          f"max={speeds.max():.3f}")
    print(f"Fraction of contacts >0.15 m/s: "
          f"{100.0 * np.mean(speeds > 0.15):.1f}%")
    print(f"Finger joint asymmetry [rad]: mean={asymmetries.mean():.4f} "
          f"median={np.median(asymmetries):.4f} p95={np.percentile(asymmetries, 95):.4f} "
          f"max={asymmetries.max():.4f}")
    print(f"Fraction of contacts with asymmetry >0.15 rad: "
          f"{100.0 * np.mean(asymmetries > 0.15):.1f}%")


if __name__ == "__main__":
    main()

"""
diag_contact_grace.py
======================
Kontakt-jakosc diagnostyka (v2.4 vs v2.3): mierzy dokladnie to, co bylo
zdiagnozowane z obserwacji uzytkownika 2026-07-20 jako wada v2.3 —
brak gracji w klikaniu (obijanie sie o zle przyciski, wysoka predkosc
kontaktu, asymetria szczek chwytaka).

Uzycie:
    python training/diag_contact_grace.py MODEL.zip --episodes 20 --seed 777

Metryki (per krok z kontaktem palec-przycisk, agregowane per epizod i globalnie):
  - obijanie (bumping): kontakt z przyciskiem != cel i JESZCZE NIEwcisnietym
    — prawdziwe obijanie sie po drodze do celu
  - zalegajacy kontakt (lingering): kontakt z przyciskiem != cel, ale JUZ
    wcisnietym — artefakt przeskoku celu po sukcesie (palec dotyka dopiero-co
    wcisnietego grzybka podczas odjazdu); raportowany OSOBNO, bo liczenie go
    jako obijania zawyzalo metryke ~13x (93% "zlych kontaktow" to zaleganie)
    i bylo prawdziwa przyczyna zapasci fine-tune v2.4
  - contact_speed: norm predkosci EEF (get_site_xvelp) w chwili kontaktu
    [m/s] — srednia, mediana, p95, max
  - joint_asymmetry: |left_inner_finger_joint - right_inner_finger_joint|
    [rad] w chwili kontaktu — miara "jedna szczeka dopchnieta do nadgarstka"
    (fizyka kp=20 przegrywajacego z momentem uderzenia, nie bug)
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
    bump_steps = 0       # kontakt z NIEwcisnietym nie-celem (prawdziwe obijanie)
    linger_steps = 0     # kontakt tylko z JUZ wcisnietymi nie-celami (odjazd)
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
    print(f"Epizody: {args.episodes}, sukcesy: {sum(ep_success)}/{args.episodes}")
    print(f"Kroki z kontaktem palec-przycisk: {contact_steps_total}")
    print(f"Obijanie o NIEwcisniety nie-cel: {bump_steps} "
          f"({100.0 * bump_steps / max(1, contact_steps_total):.1f}%)")
    print(f"Zalegajacy kontakt z wcisnietym nie-celem (odjazd po sukcesie): "
          f"{linger_steps} "
          f"({100.0 * linger_steps / max(1, contact_steps_total):.1f}%)")
    print(f"Srednia frakcja obijania per epizod: "
          f"{np.mean(ep_bump_frac):.3f}")
    print(f"Predkosc EEF przy kontakcie [m/s]: mean={speeds.mean():.3f} "
          f"median={np.median(speeds):.3f} p95={np.percentile(speeds, 95):.3f} "
          f"max={speeds.max():.3f}")
    print(f"Frakcja kontaktow >0.15 m/s: "
          f"{100.0 * np.mean(speeds > 0.15):.1f}%")
    print(f"Asymetria stawow palcow [rad]: mean={asymmetries.mean():.4f} "
          f"median={np.median(asymmetries):.4f} p95={np.percentile(asymmetries, 95):.4f} "
          f"max={asymmetries.max():.4f}")
    print(f"Frakcja kontaktow z asymetria >0.15 rad: "
          f"{100.0 * np.mean(asymmetries > 0.15):.1f}%")


if __name__ == "__main__":
    main()

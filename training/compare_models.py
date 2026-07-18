"""
Porownanie modeli na ButtonBoard-v0 — parowane epizody
======================================================
Uzycie:
    python training/compare_models.py logs/board/best_model/best_model_v4_6M.zip \
                                      logs/board/best_model/best_model.zip
    python training/compare_models.py A=sciezka1.zip B=sciezka2.zip --episodes 100
    python training/compare_models.py model.zip --episodes 50 --csv wyniki.csv

Metodologia (pod prace):
  PAROWANIE: przed KAZDYM epizodem generator losowy jest re-seedowany
  wartoscia (seed + numer_epizodu). Cala losowosc srodowiska (pozycja boardu,
  kolejnosc przyciskow, bias percepcji, szum per krok) idzie przez np.random,
  wiec epizod k ma IDENTYCZNE warunki dla kazdego modelu — lacznie z sekwencja
  szumu percepcji krok-po-kroku (liczba probek generatora na krok jest stala).
  Trajektorie roznia sie wylacznie przez akcje polityki (deterministic=True,
  fizyka MuJoCo deterministyczna).

Metryki (na epizod, potem agregaty):
  - sukces: wszystkie 5 przyciskow wcisniete >= 12 mm (epizod terminuje)
  - kroki: dlugosc epizodu (sredni czas sukcesu liczony TYLKO po sukcesach)
  - kolizja: liczba krokow z kontaktem geomu robota z plyta (przyciski sie
    nie licza — patrz ButtonBoardEnv._check_board_collision)
  - misuse: liczba krokow z kontaktem NIE-palcowej czesci robota (nadgarstek,
    ramie, obudowa grippera) z przyciskiem — v2.2 karze to jak kolizje,
    a wcisniecie bez kontaktu palcow z grzybkiem nie liczy sie do sekwencji
  - gladkosc: srednie ||a_t - a_{t-1}|| na krok (szarpanie komend, nie predkosc)
  - stuck: przekatna bounding boxa pozycji EEF z ostatnich 200 krokow < 3 cm
"""

import argparse
import csv
import os
import sys

sys.modules["tensorflow"] = None  # patrz docs 7.6 (segfault TF vs torch._dynamo)

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import robosuite as suite
from envs.button_board import ButtonBoardEnv


def flatten_obs(obs_dict):
    return np.concatenate(
        [np.asarray(v).flatten() for v in obs_dict.values()],
        dtype=np.float32,
    )


def run_episode(env, model, seed):
    """Jeden epizod o warunkach wyznaczonych przez seed; zwraca metryki."""
    np.random.seed(seed)  # parowanie: identyczne warunki dla kazdego modelu
    obs = env.reset()

    arm = env.robots[0].arms[0]
    eef_site = env.robots[0].eef_site_id[arm]

    done, t, coll_steps, misuse_steps, rate_sum = False, 0, 0, 0, 0.0
    prev_a = None
    eef_hist = []

    while not done:
        action, _ = model.predict(flatten_obs(obs), deterministic=True)
        obs, reward, done, _ = env.step(action)
        t += 1
        if env._check_board_collision():
            coll_steps += 1
        if env._check_button_misuse():
            misuse_steps += 1
        if prev_a is not None:
            rate_sum += float(np.linalg.norm(action - prev_a))
        prev_a = action
        eef_hist.append(env.sim.data.site_xpos[eef_site].copy())

    success = env._check_success()
    stuck = False
    if not success and len(eef_hist) >= 200:
        seg = np.array(eef_hist[-200:])
        stuck = float(np.linalg.norm(seg.max(axis=0) - seg.min(axis=0))) < 0.03

    return {
        "success": bool(success),
        "pressed": int(sum(env._pressed)),
        "steps": t,
        "coll_steps": coll_steps,
        "misuse_steps": misuse_steps,
        "action_rate": rate_sum / max(t - 1, 1),
        "stuck": bool(stuck),
    }


def summarize(label, rows):
    succ = np.array([r["success"] for r in rows])
    steps = np.array([r["steps"] for r in rows])
    coll = np.array([r["coll_steps"] for r in rows])
    misuse = np.array([r["misuse_steps"] for r in rows])
    rate = np.array([r["action_rate"] for r in rows])
    pressed = np.array([r["pressed"] for r in rows])
    stuck = np.array([r["stuck"] for r in rows])

    n = len(rows)
    print(f"\n=== {label} (n={n}) ===")
    print(f"sukcesy:                  {succ.sum()}/{n} ({100*succ.mean():.0f}%)")
    if succ.any():
        print(f"sredni czas sukcesu:      {steps[succ].mean():.0f} krokow "
              f"({steps[succ].mean()/20:.1f} s przy 20 Hz)")
    print(f"epizody z kolizja:        {(coll > 0).sum()}/{n}")
    print(f"sr. krokow kolizji/epizod: {coll.mean():.1f}")
    print(f"epizody z misuse przycisku: {(misuse > 0).sum()}/{n}")
    print(f"sr. krokow misuse/epizod:  {misuse.mean():.1f}")
    print(f"gladkosc sr. ||da||/krok:  {rate.mean():.4f}")
    fails = [int(((~succ) & (pressed == k)).sum()) for k in range(5)]
    print(f"porazki wg wcisnietych (0..4/5): {fails}, stuck: {stuck.sum()}")
    return succ


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+",
                   help="sciezki do modeli, opcjonalnie 'etykieta=sciezka.zip'")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--csv", type=str, default=None,
                   help="zapis metryk per epizod do pliku CSV")
    return p.parse_args()


def main():
    args = parse_args()
    from stable_baselines3 import PPO

    specs = []
    for m in args.models:
        label, _, path = m.rpartition("=")
        specs.append((label or os.path.basename(path), path or m))

    ctrl = suite.load_composite_controller_config(
        controller="BASIC", robot="Kinova3")
    env = ButtonBoardEnv(
        robots="Kinova3", controller_configs=ctrl,
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, horizon=args.horizon,
        random_order=True, board_pos_noise=0.05,
    )

    all_rows = {}
    for label, path in specs:
        print(f"\n### {label}: {path}")
        model = PPO.load(path, device="cpu")
        rows = []
        for ep in range(args.episodes):
            r = run_episode(env, model, seed=args.seed + ep)
            rows.append(r)
            print(f"  ep{ep:03d}: sukces={r['success']} {r['pressed']}/5 "
                  f"kroki={r['steps']} kolizja={r['coll_steps']} "
                  f"misuse={r['misuse_steps']} "
                  f"rate={r['action_rate']:.3f}", flush=True)
        all_rows[label] = rows

    succ_by_model = {}
    for label, rows in all_rows.items():
        succ_by_model[label] = summarize(label, rows)

    # Test istotnosci roznicy sukcesow dla dokladnie dwoch modeli
    if len(specs) == 2:
        try:
            from scipy.stats import fisher_exact
            (l1, s1), (l2, s2) = succ_by_model.items()
            table = [[int(s1.sum()), int((~s1).sum())],
                     [int(s2.sum()), int((~s2).sum())]]
            _, pval = fisher_exact(table)
            print(f"\nFisher exact ({l1} vs {l2}), sukcesy: p = {pval:.2e}")
        except ImportError:
            pass

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "epizod", "seed", "sukces", "wcisniete",
                        "kroki", "kroki_kolizji", "kroki_misuse",
                        "action_rate", "stuck"])
            for label, rows in all_rows.items():
                for ep, r in enumerate(rows):
                    w.writerow([label, ep, args.seed + ep, int(r["success"]),
                                r["pressed"], r["steps"], r["coll_steps"],
                                r["misuse_steps"],
                                f"{r['action_rate']:.5f}", int(r["stuck"])])
        print(f"CSV zapisany: {args.csv}")

    env.close()


if __name__ == "__main__":
    main()

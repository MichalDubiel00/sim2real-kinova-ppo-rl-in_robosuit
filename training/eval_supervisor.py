"""
Test akceptacyjny nadzorcy na SimIO — polityka przez PELNA petle nadzorcy
=========================================================================
Uzycie:
    python training/eval_supervisor.py --model logs/board/ppo_board_v23_12M.zip
    python training/eval_supervisor.py --model ... --no-retry --episodes 100
    python training/eval_supervisor.py --model ... --csv wyniki.csv

Polityka NIE dostaje obs z env: nadzorca buduje je sam z percepcji
(poza boardu z symulowanego AprilTag + CAD) i wlasnej maszyny stanow,
a wcisniecia wykrywa detektorem glebokosc+sila+debounce. Env sluzy tylko
za fizyke i za PRAWDE do oceny (sukces wg env, jak elektryczny board).

Warunki epizodow parowane przez seed jak w compare_models.py (pozycja
boardu, kolejnosc) — wyniki porownywalne z ewaluacja bezposrednia.
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
from supervisor import (SimIO, BoardPerception, PressDetector,
                        SimTruthDetector, TaskStateMachine, RetryManager,
                        ObsBuilder, SafetyLayer, Supervisor)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--no-retry", action="store_true",
                   help="wylacz mechanizm retry (czysta walidacja petli)")
    p.add_argument("--detector", choices=["sim", "force"], default="sim",
                   help="sim = zatrzask/wizja (SimTruthDetector); force = "
                        "sila+kinematyka (niewykonalna dla stylu v2.3 — "
                        "patrz wniosek w press_detector.py)")
    p.add_argument("--retry-timeout", type=float, default=20.0,
                   help="sekundy bez wcisniecia do odjazdu")
    p.add_argument("--force-stop", type=float, default=float("inf"),
                   help="prog force-stop [N]; inf = wylaczony (sily "
                        "kontaktowe w sim sa nierealistyczne)")
    p.add_argument("--csv", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    from stable_baselines3 import PPO

    ctrl = suite.load_composite_controller_config(
        controller="BASIC", robot="Kinova3")
    env = ButtonBoardEnv(
        robots="Kinova3", controller_configs=ctrl,
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, horizon=args.horizon,
        random_order=True, board_pos_noise=0.05,
    )
    io = SimIO(env, board_meas_err=env.percep_pos_bias)

    # Szablon kolejnosci kluczy obs — musi odpowiadac treningowi
    sample = env.reset()
    template_keys = list(sample.keys())
    builder = ObsBuilder(template_keys)

    perception = BoardPerception(
        button_faces=env.BUTTON_FACES,
        cap_offset=io.calibrate_cap_offset(),
        pos_noise=env.percep_pos_noise,
        axis_noise=env.percep_axis_noise,
    )
    detector = (SimTruthDetector(io) if args.detector == "sim"
                else PressDetector())
    task_sm = TaskStateMachine(n_buttons=env.N_BUTTONS)
    timeout = 1e9 if args.no_retry else args.retry_timeout
    retry = RetryManager(control_freq=env.control_freq, timeout=timeout)
    safety = SafetyLayer(force_stop=args.force_stop)

    model = PPO.load(args.model, device="cpu")

    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    sup = Supervisor(io, perception, detector, task_sm, retry, safety,
                     builder, policy_fn)

    # Kontrola zgodnosci wymiaru obs env vs polityka (szablon z treningu)
    flat = np.concatenate([np.asarray(v).flatten() for v in sample.values()])
    assert flat.shape[0] == model.observation_space.shape[0], \
        f"obs {flat.shape[0]} != polityka {model.observation_space.shape[0]}"
    print(f"model: {args.model} (obs dim {flat.shape[0]} OK)")
    print(f"retry: {'OFF' if args.no_retry else f'ON (timeout {args.retry_timeout}s)'}"
          f", detektor: {args.detector}")

    rows = []
    for ep in range(args.episodes):
        np.random.seed(args.seed + ep)  # parowanie warunkow jak w compare
        stats = sup.run_episode(order=None, max_steps=args.horizon)
        r = {
            "success": bool(env._check_success()),   # PRAWDA z env
            "pressed": int(sum(env._pressed)),
            "detected": stats["presses_detected"],
            "steps": stats["steps"],
            "retries": stats["retries"],
            "retreat_steps": stats["retreat_steps"],
            "force_stops": stats["force_stops"],
        }
        rows.append(r)
        print(f"  ep{ep:03d}: sukces={r['success']} {r['pressed']}/5 "
              f"(detektor {r['detected']}) kroki={r['steps']} "
              f"retry={r['retries']}", flush=True)

    succ = np.array([r["success"] for r in rows])
    steps = np.array([r["steps"] for r in rows])
    pressed = np.array([r["pressed"] for r in rows])
    detected = np.array([r["detected"] for r in rows])
    retries = np.array([r["retries"] for r in rows])
    n = len(rows)
    print(f"\n=== nadzorca ({'no-retry' if args.no_retry else 'retry'}, n={n}) ===")
    print(f"sukcesy (wg env):        {succ.sum()}/{n} ({100*succ.mean():.0f}%)")
    if succ.any():
        print(f"sredni czas sukcesu:     {steps[succ].mean():.0f} krokow "
              f"({steps[succ].mean()/20:.1f} s)")
    print(f"wcisniecia env vs detektor: {pressed.sum()} vs {detected.sum()} "
          f"(rozjazd = falszywe det. lub pominiecia)")
    print(f"epizody z retry:         {(retries > 0).sum()}/{n} "
          f"(sr. {retries.mean():.2f}/epizod)")
    fails = [int(((~succ) & (pressed == k)).sum()) for k in range(5)]
    print(f"porazki wg wcisnietych (0..4/5): {fails}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epizod", "seed", "sukces", "wcisniete", "detektor",
                        "kroki", "retry", "kroki_odjazdu", "force_stopy"])
            for ep, r in enumerate(rows):
                w.writerow([ep, args.seed + ep, int(r["success"]),
                            r["pressed"], r["detected"], r["steps"],
                            r["retries"], r["retreat_steps"],
                            r["force_stops"]])
        print(f"CSV zapisany: {args.csv}")

    env.close()


if __name__ == "__main__":
    main()

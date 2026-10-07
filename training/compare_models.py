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
  - misuse: liczba krokow z kontaktem OGNIWA RAMIENIA (robot0_*) z przyciskiem
    — v2.3 karze to jak kolizje (kontakt chwytaka jest dozwolony, bo
    towarzyszy poprawnym docisnieciom palcami); wcisniecie bez kontaktu
    palcow z grzybkiem nadal nie liczy sie do sekwencji
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
from envs.controller_scale import with_osc_output_max

# Limity predkosci stawow Gen3 6DOF — ros2_kortex, plik
# kortex_description/arms/gen3/6dof/config/joint_limits.yaml:
# maximum_velocities = 0.8727 rad/s dla WSZYSTKICH szesciu stawow (= 50 st./s,
# zgodne z karta katalogowa). NIE mylic z wartosciami 1.3963/1.2218 z PR #241,
# ktory zmienial tagi <velocity> w gen3_macro.xacro — patrz docs 6.31.
HW_JOINT_VEL = [0.8727] * 6
# Instrukcja Gen3 R07 (docs 6.37): limity ogolne tab. 46, tryb twist = skalowanie proporcjonalne
KINOVA_JOINT_VEL = [1.39, 1.39, 1.39, 1.22, 1.22, 1.22]
KINOVA_TWIST = (0.5, 0.8727)
# Instrukcja Gen3 R07, tab. 48 (Gen3 6 DoF, TWARDE limity przyspieszen — obowiazuja
# zawsze). Tab. 49 (1.0 rad/s^2 dla j1-3) dotyczy TYLKO angular joystick i joint
# trajectory; nasz tor to Cartesian joystick (twist), ktoremu instrukcja nakłada
# wylacznie limity pozycji i predkosci (s. 100). Docs 6.44.
HW_JOINT_ACC = [5.2, 5.2, 5.2, 10.0, 10.0, 10.0]


def flatten_obs(obs_dict):
    return np.concatenate(
        [np.asarray(v).flatten() for v in obs_dict.values()],
        dtype=np.float32,
    )


def run_episode(env, model, seed, action_ema=None):
    """Jeden epizod o warunkach wyznaczonych przez seed; zwraca metryki.

    action_ema: stala filtru dolnoprzepustowego na AKCJI (0 < alpha <= 1).
    a_t = alpha * a_polityki + (1 - alpha) * a_{t-1}. alpha = 1 lub None = bez
    filtru. Motywacja sim2real: polityka komenderuje bang-bang (||da|| 1.335 na
    krok przy 71 % nasyceniu), co na sprzecie oznacza szarpanie komendy o dwie
    trzecie zakresu co 50 ms i przekraczanie limitow PRZYSPIESZEN. Filtr jest
    zmiana po stronie WDROZENIA, nie treningu — dlatego nie podpada pod zakaz
    naprawiania zachowania rewardem (6.20).
    """
    np.random.seed(seed)  # parowanie: identyczne warunki dla kazdego modelu
    obs = env.reset()

    arm = env.robots[0].arms[0]
    eef_site = env.robots[0].eef_site_id[arm]

    done, t, coll_steps, misuse_steps, rate_sum = False, 0, 0, 0, 0.0
    spot_steps = 0
    prev_a = None
    eef_hist = []

    a_filt = None
    while not done:
        action, _ = model.predict(flatten_obs(obs), deterministic=True)
        if action_ema is not None and action_ema < 1.0:
            a_filt = (action.astype(np.float64) if a_filt is None
                      else action_ema * action + (1.0 - action_ema) * a_filt)
            action = a_filt.astype(np.float32)
        obs, reward, done, _ = env.step(action)
        t += 1
        if env._check_board_collision():
            coll_steps += 1
        if env._check_button_misuse():
            misuse_steps += 1
        if env._check_spot_collision():
            spot_steps += 1
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
        "pressed_mask": [bool(p) for p in env._pressed],
        "steps": t,
        "coll_steps": coll_steps,
        "spot_steps": spot_steps,
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
    spot = np.array([r.get("spot_steps", 0) for r in rows])
    if spot.sum() or any("spot_steps" in r for r in rows):
        print(f"epizody z kontaktem ze Spotem: {(spot > 0).sum()}/{n}")
        print(f"sr. krokow kontaktu ze Spotem: {spot.mean():.1f}")
    print(f"epizody z misuse przycisku: {(misuse > 0).sum()}/{n}")
    print(f"sr. krokow misuse/epizod:  {misuse.mean():.1f}")
    print(f"gladkosc sr. ||da||/krok:  {rate.mean():.4f}")
    fails = [int(((~succ) & (pressed == k)).sum()) for k in range(5)]
    print(f"porazki wg wcisnietych (0..4/5): {fails}, stuck: {stuck.sum()}")
    per_btn = np.array([r["pressed_mask"] for r in rows])  # (n, 5)
    print("wcisniecia per przycisk:   "
          + ", ".join(f"btn{i}: {per_btn[:, i].sum()}/{n}" for i in range(per_btn.shape[1])))
    return succ


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+",
                   help="sciezki do modeli, opcjonalnie 'etykieta=sciezka.zip'")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--button-stiffness", type=float, default=None,
                   help="sztywnosc sprezyny grzybka [N/m]; brak = 500 z button.xml (~6 N "
                        "przy 12 mm). Typowy E-stop 20-40 N -> 1700-3300 N/m (docs 6.33)")
    p.add_argument("--pos-max", type=float, default=None,
                   help="output_max translacji OSC [m/krok]; MUSI byc zgodne z treningiem "
                        "modelu (domyslnie 0.05; docs 6.36)")
    p.add_argument("--rot-max", type=float, default=None,
                   help="output_max obrotu OSC [rad/krok]; MUSI byc zgodne z treningiem "
                        "modelu (domyslnie 0.5; docs 6.35)")
    p.add_argument("--estop-config", type=str, default=None,
                   help="YAML treningu: bierze z niego button_estop_profile i "
                        "activation_depth (np. E-stop 20 N / 6 mm). Wyklucza --estop")
    p.add_argument("--estop", action="store_true",
                   help="nieliniowy zatrzaskowy model E-stopu "
                        "ButtonBoardEnv.ESTOP_PROFILE_EXAMPLE (15->35 N do 3.5 mm, "
                        "spadek do 5 N, skok 5 mm, zatrzask; docs 6.34). Bez "
                        "--activation-depth prog = punkt rozwarcia stykow (3.8 mm)")
    p.add_argument("--kinova-limits", action="store_true",
                   help="limity wg instrukcji Gen3 R07 dla trybu twist: stawy 1.39/1.22 rad/s "
                        "skalowane proporcjonalnie + narzedzie 0.5 m/s, 0.8727 rad/s (docs 6.37)")
    p.add_argument("--hw-limits", action="store_true",
                   help="nalozy limity predkosci stawow z firmware Gen3 "
                        "(ros2_kortex joint_limits.yaml): 0.8727 rad/s wszystkie stawy")
    p.add_argument("--gain", type=float, default=None,
                   help="STALE wzmocnienie mechanizmu na wszystkich 6 osiach "
                        "(1.0 = jak w treningu). Symuluje rozbieznosc sim2sim "
                        "zmierzona w Webots: 0.32 transl., 0.10 obrot (docs 6.30).")
    p.add_argument("--robot", type=str, default="Kinova3",
                   help="Kinova3 (7DOF, stock) albo Kinova3_6DOF (uczelniany); "
                        "modele 6DOF sa NIEkompatybilne z 7DOF (obs 120 vs 128)")
    p.add_argument("--activation-depth", type=float, default=None,
                   help="prog wcisniecia w metrach; domyslnie z env (0.012), "
                        "modele 6DOF etapu 1 trenowane na 0.010")
    p.add_argument("--no-finger-gate", action="store_true",
                   help="wylacz bramke palcowa. MUSI byc zgodne z trybem, w ktorym "
                        "model trenowal — etap 1 kurikulum (`depress`, `hwlim_s1`) ma "
                        "finger_press_gate=false, wiec mierzony z wlaczona bramka "
                        "dostaje 0/40 mimo poprawnego dojazdu do przycisku")
    p.add_argument("--free-gripper", action="store_true",
                   help="chwytak sterowany przez polityke (force_gripper_closed"
                        "=False). MUSI byc zgodne z trybem, w ktorym model byl "
                        "trenowany — niezgodnosc zmienia zadanie (pomiar A/B: "
                        "82%% -> 50%% na tym samym modelu, docs 6.15)")
    p.add_argument("--acc-limits", action="store_true",
                   help="TWARDE limity przyspieszen Gen3 6DOF wg instrukcji R07 tab. 48: "
                        "[5.2 x3, 10.0 x3] rad/s^2. UWAGA: 1.0 rad/s^2 z tab. 49 dotyczy "
                        "TYLKO angular joystick/joint trajectory, NIE toru twist (docs 6.44)")
    p.add_argument("--acc-mode", choices=["clip", "scale"], default="clip",
                   help="semantyka przycinania przyspieszen (docs 6.50). clip = niezaleznie "
                        "na kazdym stawie (zmienia kierunek ruchu). scale = caly przyrost "
                        "predkosci skalowany jednym wspolczynnikiem, kierunek zachowany — "
                        "tak wg instrukcji Gen3 R07 robi firmware w trybie twist. Obie wersje "
                        "trzymaja TE SAMA obwiednie; roznica jest w kierunku po przycieciu")
    p.add_argument("--acc-ref", choices=["qvel", "cmd"], default="qvel",
                   help="skala, w ktorej dziala limit przyspieszen (docs 6.48). "
                        "qvel = wzgledem zmierzonej predkosci z poprzedniego podkroku "
                        "(dotychczasowe; budzet zjada tetnienie OSC, przy a=0 mediana "
                        "40 rad/s^2 na j3 mimo stojacego ramienia). cmd = wzgledem "
                        "wlasnej poprzedniej komendy, czyli nachylenie ZADANEGO profilu "
                        "predkosci — to, co ogranicza firmware")
    p.add_argument("--base-x-offset", type=float, default=None, metavar="M",
                   help="odleglosc postoju Spota: przesuniecie bazy ramienia wzdluz X. "
                        "Domyslne env to -0.10; biegi od docs 6.42 uzywaja -0.34. "
                        "BEZ tego ocena modelu uczonego na -0.34 szla by w INNEJ scenie")
    p.add_argument("--init-qpos", type=str, default=None, metavar="q1,..,q6",
                   help="poza startowa jako 6 katow po przecinku; brak = domyslna "
                        "(chwytak pionowo w dol, kamera nadgarstkowa nie widzi tablicy)")
    p.add_argument("--obs-acc-saturation", action="store_true",
                   help="dodaj sygnal nasycenia do obserwacji (obs 120 -> 132, docs 6.47). Modele sprzed 2026-10-07 maja wejscie 120 i NIE wczytaja sie z ta flaga")
    p.add_argument("--action-rate-limit", type=float, default=None, metavar="D",
                   help="OGRANICZENIE tempa komendy: |a_t - a_{t-1}| <= D na skladowa, "
                        "nakladane przed kontrolerem (docs 6.47). Nie kara — niezbywalne")
    p.add_argument("--approach-speed-cost", type=float, default=0.0,
                   help="kara za predkosc zblizania ponad v_dop = dystans/0.3 s (docs 6.45); "
                        "w EWALUACJI zwykle 0 — zmienia nagrode, nie zachowanie")
    p.add_argument("--spot-body", action="store_true",
                   help="wstaw bryle korpusu Spota pod ramieniem (zmierzona z Webotsa, "
                        "docs 6.41). W treningu tego NIE BYLO — wlaczenie zmienia zadanie")
    p.add_argument("--spot-front-cam", action="store_true",
                   help="wstaw przednia kamere Spota (obrys Kinecta z Webotsa jako bufor): "
                        "listwa 28 cm wszerz, 21-27 cm przed podstawa, na jej wysokosci")
    p.add_argument("--action-ema", type=float, default=None, metavar="ALPHA",
                   help="filtr dolnoprzepustowy na akcji: a_t = alpha*a + (1-alpha)*a_{t-1}. "
                        "1.0 lub brak = bez filtru. Zmiana po stronie wdrozenia — "
                        "lagodzi bang-bang (||da|| 1.335/krok) przed wyjsciem na sprzet")
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

    if args.init_qpos:
        args.init_qpos = [float(x) for x in args.init_qpos.split(",")]
    estop_prof = ButtonBoardEnv.ESTOP_PROFILE_EXAMPLE if args.estop else None
    if args.estop_config:
        if args.estop:
            raise SystemExit("--estop i --estop-config wykluczaja sie")
        import yaml
        _c = yaml.safe_load(open(args.estop_config))
        estop_prof = _c["button_estop_profile"]
        if args.activation_depth is None:
            args.activation_depth = _c.get("activation_depth")
        print(f"[estop] z {args.estop_config}: {estop_prof}, prog {args.activation_depth}")
    ctrl = with_osc_output_max(suite.load_composite_controller_config(
        controller="BASIC", robot=args.robot), args.pos_max, args.rot_max)
    env = ButtonBoardEnv(
        robots=args.robot, controller_configs=ctrl,
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, horizon=args.horizon,
        # board_pos_noise / base_z_noise / percep_* CELOWO nie podawane:
        # biora domyslne z ButtonBoardEnv, zeby pomiar szedl w TYCH SAMYCH
        # warunkach co trening. Do 2026-09-07 bylo tu zaszyte 0.05, ktore
        # nie zostalo zaktualizowane przy podniesieniu szumu 01.09 (docs 6.24).
        random_order=True,
        activation_depth=args.activation_depth,
        force_gripper_closed=not args.free_gripper,
        finger_press_gate=not args.no_finger_gate,
        # zakres zdegenerowany do punktu = wzmocnienie STALE, nie losowane
        gain_noise_lin=((args.gain, args.gain) if args.gain is not None else None),
        gain_noise_rot=((args.gain, args.gain) if args.gain is not None else None),
        joint_acc_limits=(HW_JOINT_ACC if args.acc_limits else None),
        joint_acc_limit_ref=args.acc_ref,
        joint_acc_limit_mode=args.acc_mode,
        joint_vel_limits=(KINOVA_JOINT_VEL if args.kinova_limits
                          else HW_JOINT_VEL if args.hw_limits else None),
        joint_vel_limit_mode=("scale" if args.kinova_limits else "clip"),
        eef_twist_limits=(KINOVA_TWIST if args.kinova_limits else None),
        **({"base_x_offset": args.base_x_offset} if args.base_x_offset is not None else {}),
        init_qpos=args.init_qpos,
        obs_acc_saturation=args.obs_acc_saturation,
        action_rate_limit=args.action_rate_limit,
        approach_speed_cost=args.approach_speed_cost,
        spot_body=args.spot_body,
        spot_front_cam=args.spot_front_cam,
        button_stiffness=args.button_stiffness,
        button_estop_profile=estop_prof,
    )

    all_rows = {}
    for label, path in specs:
        print(f"\n### {label}: {path}")
        model = PPO.load(path, device="cpu")
        rows = []
        for ep in range(args.episodes):
            r = run_episode(env, model, seed=args.seed + ep, action_ema=args.action_ema)
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
            # kroki_spot na KONCU, zeby stare skrypty czytajace po indeksie
            # (paired_candidates.py, analizy z 6.25-6.40) dalej dzialaly
            w.writerow(["model", "epizod", "seed", "sukces", "wcisniete",
                        "kroki", "kroki_kolizji", "kroki_misuse",
                        "action_rate", "stuck", "kroki_spot"])
            for label, rows in all_rows.items():
                for ep, r in enumerate(rows):
                    w.writerow([label, ep, args.seed + ep, int(r["success"]),
                                r["pressed"], r["steps"], r["coll_steps"],
                                r["misuse_steps"],
                                f"{r['action_rate']:.5f}", int(r["stuck"]),
                                r.get("spot_steps", 0)])
        print(f"CSV zapisany: {args.csv}")

    env.close()


if __name__ == "__main__":
    main()

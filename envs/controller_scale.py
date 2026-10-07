"""Skala obrotu OSC_POSE (docs 6.35).

output_max obrotu mowi, ile radianow obrotu celu oznacza akcja 1.0 w jednym kroku
sterowania. Domyslne 0.5 rad kaze sterownikowi zadac predkosci stawow wielokrotnie
powyzej limitu firmware Gen3 (0.8727 rad/s); limit obcina kazdy staw osobno, co
zmienia KIERUNEK ruchu, nie tylko tempo. Mniejszy output_max skaluje tez szum
eksploracji PPO, ktory przy std 1.0 sam przekraczal limit.

UWAGA sim2real: most (ros2_bridge/policy_bridge.py, ROT_SCALE) musi uzywac tej
samej wartosci, inaczej akcja 1.0 znaczy co innego w symulacji i na sprzecie.
"""
import copy

import numpy as np


def with_osc_output_max(ctrl_cfg, pos_max=None, rot_max=None):
    """Kopia configu kontrolera z output_max/min = +-pos_max (translacja, m/krok) i
    +-rot_max (obrot, rad/krok) na czesciach OSC_POSE. None = ta os bez zmian.
    Translacja: docs 6.36 — nasycona translacja 0.05 m/krok przekraczala limit stawow
    w 73-90 % prob niezaleznie od skali obrotu."""
    if pos_max is None and rot_max is None:
        return ctrl_cfg
    cfg = copy.deepcopy(ctrl_cfg)
    n = 0
    for part in cfg.get("body_parts", {}).values():
        if part.get("type") == "OSC_POSE":
            om = [float(x) for x in np.asarray(part["output_max"], dtype=float)]
            if pos_max is not None:
                om[:3] = [float(pos_max)] * 3
            if rot_max is not None:
                om[3:] = [float(rot_max)] * 3
            part["output_max"] = om
            part["output_min"] = [-x for x in om]
            n += 1
    if n == 0:
        raise ValueError("brak czesci OSC_POSE w configu kontrolera")
    return cfg


def with_osc_rot_output_max(ctrl_cfg, rot_max):
    """Zgodnosc wstecz: tylko obrot."""
    return with_osc_output_max(ctrl_cfg, rot_max=rot_max)

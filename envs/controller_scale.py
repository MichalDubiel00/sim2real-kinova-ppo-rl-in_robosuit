"""OSC_POSE rotation scale (docs 6.35).

Rotation output_max = target rotation per step at action 1.0. The default 0.5 rad
demands joint speeds far above the Gen3 limit (0.8727 rad/s), and per-joint clipping
changes the motion DIRECTION. The bridge must use the same value.
"""
import copy

import numpy as np


def with_osc_output_max(ctrl_cfg, pos_max=None, rot_max=None):
    """Copy of the controller config with output_max/min = +-pos_max (m/step) and
    +-rot_max (rad/step) on OSC_POSE parts; None = unchanged. Translation: docs 6.36
    (0.05 m/step exceeded joint limits in 73-90 % of trials)."""
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
        raise ValueError("no OSC_POSE part in the controller config")
    return cfg


def with_osc_rot_output_max(ctrl_cfg, rot_max):
    """Backward compatibility: rotation only."""
    return with_osc_output_max(ctrl_cfg, rot_max=rot_max)

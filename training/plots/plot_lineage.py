"""Learning curve of the hardware-model lineage, from scratch (sc1) to arm_d.

Reads TensorBoard events (logs/tensorboard and logs/tensorboard_chmura, see
cloud/tb_pull.sh) and plots the mean evaluation episode length as a fraction of the
horizon: episodes end early only on success, so this is comparable across stages with
different rewards and horizons. Each run is cut where the next stage resumed from it.

    python training/plots/plot_lineage.py --out docs/img/learning_curve_lineage.png
"""
import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

T, C = "logs/tensorboard/", "logs/tensorboard_chmura/"
# (label, event dirs, horizon, first step [M], last step [M]): the resumed lineage of arm_d
STAGES = [
    ("sc1", [T + "sc1_loose_clamp_2"], 1000, 0.0, 4.01),
    ("ft_sc1_v23", [T + "ft_sc1_v23_0", T + "ft_sc1_v23_long_0"], 1000, 4.01, 15.91),
    ("estop_curr", [T + "ppo_board_6dof_estop_curr_0"], 1500, 15.91, 16.01),
    ("kinova_ft", [T + "ppo_board_6dof_kinova_ft_0"], 2000, 16.01, 17.81),
    ("kinova_ft_lr", [T + "ppo_board_6dof_kinova_ft_lr_0"], 2000, 17.81, 23.82),
    ("kinova_ft_lr2", [T + "ppo_board_6dof_kinova_ft_lr2_0"], 2000, 23.82, 28.02),
    ("e1", [T + "ppo_board_6dof_spot_e1_0"], 2000, 28.02, 30.02),
    ("e1b", [T + "ppo_board_6dof_spot_e1b_0"], 2000, 30.02, 38.92),
    ("e2", [T + "ppo_board_6dof_spot_e2_0"], 2000, 38.92, 42.22),
    ("e2b", [T + "ppo_board_6dof_spot_e2b_0"], 2000, 42.22, 50.22),
    ("e3", [T + "ppo_board_6dof_spot_e3_0"], 2000, 50.22, 58.12),
    ("e4", [T + "ppo_board_6dof_spot_e4_0"], 2000, 58.12, 66.12),
    ("e5", [T + "ppo_board_6dof_spot_e5_0"], 2000, 66.12, 74.12),
    ("e6a", [T + "ppo_board_6dof_spot_e6a_0"], 2000, 74.12, 78.12),
    ("e6ab", [T + "ppo_board_6dof_spot_e6ab_0"], 2000, 78.12, 86.12),
    ("arm_sb", [C + "arm_sb/arm_sb_0"], 2000, 86.12, 90.12),
    ("arm_d", [C + "arm_d/arm_d_0"], 2000, 90.12, 98.13),
]
# final paired evaluations, n = 100, h = 2000 (docs 6.40, 6.42, 6.50, 6.51, 6.55)
RESULTS = [(28.02, "92 %"), (50.22, "100 %"), (86.12, "100 %\nclip"),
           (90.12, "100 %"), (98.12, "98 %")]
# environment versions (docs/thesis_notes_en.md, section 3.7)
ENVS = [(0.0, 15.91, "E-C\nWebots geometry,\nperception noise"),
        (15.91, 28.02, "E-D\nGen3 limits,\nE-stop 20 N"),
        (28.02, 58.12, "E-E\nSpot body, stand-off,\nstart pose"),
        (58.12, 86.12, "E-F\nacc. limits\n(clip)"),
        (86.12, 90.12, "E-G\nscale"),
        (90.12, 98.13, "E-H\ndead\ntime")]


def scalars(dirs, tag):
    s, v = [], []
    for d in dirs:
        for path in glob.glob(d):
            ea = EventAccumulator(path, size_guidance={"scalars": 0})
            ea.Reload()
            if tag in ea.Tags()["scalars"]:
                for e in ea.Scalars(tag):
                    s.append(e.step / 1e6); v.append(e.value)
    o = np.argsort(s)
    return np.array(s)[o], np.array(v)[o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/img/learning_curve_lineage.png")
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(13, 4.8))
    shades = ["#f3f3f3", "#e3ecf7"]
    for i, (x0, x1, name) in enumerate(ENVS):
        ax.axvspan(x0, x1, color=shades[i % 2], zorder=0)
        ax.text((x0 + x1) / 2, 1.185, name, ha="center", va="top", fontsize=8)
    for k, (label, dirs, h, x0, x1) in enumerate(STAGES):
        for tag, kw in (("rollout/ep_len_mean", dict(color="#9db4d0", lw=0.8)),
                        ("eval/mean_ep_length", dict(color="#1f4e8c", lw=1.6, marker=".", ms=3))):
            s, v = scalars(dirs, tag)
            m = (s >= x0) & (s <= x1 + 0.01)
            ax.plot(s[m], v[m] / h, label=None, **kw)
        ax.axvline(x0, color="#bbbbbb", lw=0.5, zorder=1)
    for x, txt in RESULTS:
        ax.plot([x], [0.015], marker="v", color="#c0392b", ms=6, zorder=5)
        ax.text(x, 0.045, txt, ha="center", va="bottom", fontsize=7, color="#c0392b", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.plot([], [], color="#1f4e8c", lw=1.6, marker=".", label="evaluation (deterministic)")
    ax.plot([], [], color="#9db4d0", lw=0.8, label="training rollouts")
    ax.plot([], [], marker="v", color="#c0392b", ls="", label="paired evaluation, n = 100")
    ax.set_xlim(0, 98.2)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("environment steps [millions]")
    ax.set_ylabel("mean episode length / horizon\n(1.0 = no successes)")
    ax.set_title("Hardware-model lineage: scratch -> arm_d (each stage resumed from the previous one)",
                 fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.80), fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print("saved", a.out)


if __name__ == "__main__":
    main()

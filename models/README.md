# Models: scene-contract manifest

**A checkpoint without its scene contract is a trap.** It loads **without error** in a changed
scene and the only symptom is a low score; measurements were lost this way before
`compare_models.py` got `--base-x-offset` (docs 6.42). Every entry below therefore gives the
flags its number was measured with, **n** and the **horizon**. A number without these three
means nothing.

Files are in `logs/` (outside git) and backed up in `gs://sim2real_robosuite/backup/`.
Scene values are taken from the configs, not from memory.

## HARDWARE MODEL

### `arm_d/checkpoints/ppo_board_98123552_steps.zip`
- config `training/configs/arm_d.yaml` = `arm_sb` + 8M with loop dead time
  `action_delay_steps: [0, 2]` drawn per episode; h=2000, n=100, seed 1234, `scale` mode (docs 6.55)
- **d=0: 98/100, collisions 3.46 (median 0.0) | d=1: 99/100, collisions 4.90 |
  d=2: 98/100, collisions 18.74** (d = action delay in 50 ms steps)
- scene **identical to `arm_sb` and e6ab** (below); the only evaluation difference is
  `--action-delay`
- copies: `gs://sim2real_robosuite/arms/arm_d/` and `backup/decyzyjne/`

```bash
START_POSE="0.0,1.4692,-2.2410,3.1403,-2.0200,3.1416"
python training/compare_models.py "arm_d=logs/board_6dof/arm_d/checkpoints/ppo_board_98123552_steps.zip" \
  --robot Kinova3_6DOF --episodes 100 --seed 1234 --horizon 2000 \
  --kinova-limits --acc-limits --acc-mode scale --action-delay 0 \
  --estop-config training/configs/ppo_board_6dof_estop_hw.yaml \
  --spot-body --spot-front-cam --base-x-offset=-0.34 --init-qpos="$START_POSE"
```

**Fallback: `arm_sb/checkpoints/ppo_board_90123552_steps.zip`**: d=0 100/100, collisions
6.98; d=2 93/100, collisions 30.68 (n=100, h=2000, scale). Use it only if the dead time
measured on hardware in wall-clock time is persistently < 50 ms without a tail (docs 6.55).

## Previous hardware model (until 2026-10-09)

### `spot_e6ab/checkpoints/ppo_board_86123552_steps.zip`
- config `ppo_board_6dof_spot_e6ab.yaml`, h=2000, n=100
- `--acc-mode clip`: **100/100 [96.3-100 %], 15.2 s, collisions 7.0 (median 6), misuse 0.0,
  every button 100/100, 0 stuck** (docs 6.50)
- ⚠ in the more faithful `--acc-mode scale` (how the firmware clips), n=100: **99/100, 16.1 s,
  collisions 14.65 (median 6.5)** (docs 6.51). Quote **14.65** for hardware, not 7.0
- scene: Spot body + camera, `base_x_offset -0.34`,
  `init_qpos [0.0, 1.4692, -2.2410, 3.1403, -2.0200, 3.1416]`, velocities [1.39x3, 1.22x3]
  scale mode, accelerations [5.2x3, 10.0x3], twist [0.5, 0.8727], E-stop (6 mm stroke, 20 N,
  latch), `activation_depth 0.0059`

```bash
python training/compare_models.py "e6ab=logs/board_6dof/spot_e6ab/checkpoints/ppo_board_86123552_steps.zip" \
  --robot Kinova3_6DOF --episodes 100 --seed 1234 --horizon 2000 \
  --kinova-limits --acc-limits --acc-mode scale \
  --estop-config training/configs/ppo_board_6dof_estop_hw.yaml \
  --spot-body --spot-front-cam --base-x-offset=-0.34 --init-qpos="$START_POSE"
```

## Earlier models, each in a DIFFERENT scene

Compare only within one scene. Comparing numbers across rows of this table is an error
(made twice already, docs 6.42 and 6.44).

| model | scene | result |
|---|---|---|
| `spot_e6a/.../78123552` | as e6ab | 98/100 [93.0-99.4], 15.6 s, collisions 8.7, misuse 0.7, n=100 |
| `spot_e5/.../74123552` | as e6ab | 38/40 (95 %), 17.5 s, collisions 20.9 (med. 9), misuse 0.2, n=40 |
| `spot_e2b/.../50223552` | as e6ab but **WITHOUT acceleration limits** | 100/100 [96.3-100], 7.4 s, collisions 0.2, n=100 |
| `kinova_ft_lr2/.../28023552` | **WITHOUT Spot body**, `base_x -0.10`, default start pose, no acceleration limits | 92/100 [85-96], 23.9 s, collisions 2.3 (med. 1), n=100 |
| `sweep/ft_sc1_v23_long/ppo_board_final.zip` | **DIFFERENT TASK** (below) | 76/100 @h=1500, 28.3 s, collisions 7.9 |

### ⚠ `ft_sc1_v23_long` measures a different task, not only a different scene

Its config has `activation_depth 0.012` and **no E-stop profile**: a linear 500 N/m button
with a 12 mm threshold, the model from before the datasheet fix (docs 6.33: the real button
has a 5 mm stroke and 25 N). The config also has `horizon: 1000`, while 76/100 is reported at
h=1500. This model is **not** comparable to any row above under any flags; it only records
a stage.

## What the bucket backup contains

- `backup/decyzyjne/`: models from this manifest (arm_d and arm_sb added 2026-10-15)
- `backup/best_model/` and `backup/final/`: one per run (45 + 29 files)
- `backup/eval/`: evaluation CSVs (raw data behind the numbers in the docs)
- `backup/tensorboard/`: training curves

**Not included:** 2319 intermediate checkpoints (5.9 GB), reproducible and without
evidential value.

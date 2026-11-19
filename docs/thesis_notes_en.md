# Thesis notes (EN): RL button pressing with a Kinova Gen3 6DOF on Spot, sim2real

Condensed English companion to `docs/project_documentation_pl.md` (the full Polish lab
journal, ~7400 lines). This file keeps what a thesis needs: the problem, the design,
results with their exact conditions, negative results, lessons and open issues.
Section references like *(PL 6.25)* point to the journal for raw tables and logs.
State as of 2026-10-19.

**Reporting rule used throughout.** Every success rate is given with its horizon (h, in
control steps at 20 Hz; h = 2000 is 100 s) and its environment version (E-A ... E-H,
section 3.7). Models load without error after geometry changes; the only symptom of a
mismatched evaluation is a low score, so a number without its environment is meaningless.

---

## 1. Problem

- **Task.** RoboCup Rescue "omniboard": press five emergency-stop buttons mounted on a
  cross-shaped panel (front face + four 45-degree bevels, cross rolled 45 degrees on the
  wall), in an order given by an operator.
- **Robot.** Kinova Gen3 **6DOF** (the lab variant, not the stock 7DOF of robosuite) with a
  Robotiq 2F-85 gripper, mounted on a Boston Dynamics Spot.
- **Approach.** PPO policy trained in robosuite/MuJoCo; deterministic supervisor around it on
  hardware; transfer through ROS 2 (`ros2_kortex`).
- **Core design principle.** The policy only sees quantities reproducible on the robot.
  Target pose comes from perception (ArUco), never from privileged simulator state. Task
  logic (target order, press detection) and safety live outside the policy.

## 2. Software and hardware stack

| component | version / note |
|---|---|
| robosuite | 1.5.1 (OSC_POSE inside the BASIC composite controller) |
| MuJoCo | 3.3.7 |
| stable-baselines3 | 2.3.2, PPO, `MlpPolicy` [256, 256] |
| gymnasium / numpy / PyTorch | 0.29.1 / 1.26.4 / 2.12 |
| ROS 2 | Humble; `ros2_kortex` (Kinova upstream) on hardware |
| Webots | R2025a, MASKOR `webots_ros2_spot` branch `kinova_gen3` (sim2sim only) |
| training hardware | laptop i7-9750H 6C/12T, RTX 2060, 15 GB RAM; GCP `e2-standard-8` spot VMs |

Throughput: MuJoCo on CPU is the bottleneck (GPU ~1 %). ~150-290 env steps/s on the laptop
with 8 `SubprocVecEnv` workers, 164-180 on the cloud VMs. A 4M-step fine-tune is ~4-6.5 h.

## 3. Simulation environment (`envs/button_board.py`)

### 3.1 Scene and robot

- Arm base on a rigid mount at the measured Spot height; the board's back face on a wall.
- Final geometry (from the MASKOR Webots model, measured via TF, PL 6.24): board centre
  z = 0.64 m, arm base z = 0.62 m, socket radius 0.107 m, cross rolled 45 degrees.
- Final stand-off and start pose (PL 6.42): `base_x_offset = -0.34` (base 0.763 m from the
  wall face), `init_qpos = [0, 1.4692, -2.2410, 3.1403, -2.0200, 3.1416]`, chosen for
  manipulability (sqrt(det J J^T) 0.050 vs 0.023 at -0.18), joint-limit margin (0.389 rad),
  wrist-camera view of the board (3/5 buttons in frame, align 8.4 degrees) and OSC stability
  (drift 0.017 rad in 30 zero-action steps). Access to the board *improves* with distance:
  close up the arm folds near singularities.
- Spot body (16 boxes of `Spot.proto`, rotations included) and the front Kinect bar as
  collision obstacles (PL 6.42).
- 6DOF model adapted from goldengeko/robosuite_training_examples; two fixes were required:
  `init_qpos` copied from 7DOF without remapping pitch joints, and missing
  `conaffinity="0"` causing a permanent phantom base-shoulder contact (-14.4 mm) during the
  first 5.2M steps of 6DOF training (PL 6.7, 6.9).

### 3.2 Action and control

- 7-D action in [-1, 1]: 6-D OSC_POSE delta in the base frame (+ gripper, overridden to
  "closed"). Default output_max 0.05 m / 0.5 rad per step; 20 Hz control, 500 Hz physics.
- In robosuite the realised TCP velocity is a first-order filter of the action, not a gain:
  v[t+1] = alpha v[t] + beta a[t], R^2 0.87-0.97 vs 0.05-0.71 for v = k a (2484 steps of
  `arm_d`, PL 6.56). Translation time constant ~0.5 s, rotation ~0.16 s; realised speeds sit
  on the twist caps (p95 0.50 m/s, 0.87 rad/s).

### 3.3 Observation (120-D for 6DOF)

60 unique values duplicated by robosuite's `*-state` aggregates: joint pos/cos/sin/vel (24),
`eef_pos` (absolute, world frame), `eef_quat`, `eef_quat_site`, gripper qpos/qvel (12),
`target_relative_pos`, `target_distance`, `target_onehot` (5), `target_press_axis`,
`eef_align_with_target`. Contract in `envs/obs_contract.json`. Button depressions are
deliberately absent (the real board is not wired).

Pitfalls found during transfer (PL 6.26-6.27, 6.57):
- `eef_pos` is absolute, not base-relative; `eef_pos` comes from a site and `eef_quat` from a
  body (robosuite issue #298), with different quaternion sign conventions.
- `eef_pos`/`eef_quat` lag `joint_pos` by one 2 ms physics substep (positions computed before
  integration); the bridge reproduces this artifact.
- The gripper block (12 of 60 values) moves the policy disproportionately: a sign/statistic
  error there changed |a| from 0.94 to 0.45.

### 3.4 Reward (final = v2.3)

```
r = 10 (d_prev - d)                         potential-based approach shaping
  + 0.05 exp(-8 d) + dense_align_coef exp(-3 theta)
  + 25 [target pressed] + 100 [all five]    episode terminates on success
  - collision_cost [arm/board or wall contact] - 0.01 ||a||^2 - action_rate_cost ||a_t - a_{t-1}||^2
r_norm = r / 225
```
A press counts only with a gripper finger on the cap (finger gate, v2.2+); arm-link contact
with a button is penalised like a collision (v2.3). Optional terms used in curricula: depth
shaping `depress_shaping_k` with reward clipping at activation + 1 mm, Spot-body contact cost.
Reward always uses ground truth; noise exists only in observations.

### 3.5 Button model

- Original: linear spring 500 N/m, 15 mm travel, activation at 12 mm (~6 N). Arbitrary; the
  "realistic" label had no source (PL 6.33).
- Datasheet E-stop: 5 mm stroke, 25 N, latching. The sim was 4x too soft and 2.4x too deep.
- Final model (PL 6.34): non-linear latching profile, 10 -> 20 N up to 4 mm, drop to 5 N,
  6 mm stroke, latch = press (activation 5.9 mm), force via `qfrc_applied`, hard kinematic
  end stops (MuJoCo soft limits yielded 8 mm under a 15 N preload).

### 3.6 Domain randomisation and fidelity layers

| item | value | type |
|---|---|---|
| board position noise | +-6 cm (Y, Z) per episode | randomisation |
| perception bias / per-step noise | +-15 mm / 5 mm | randomisation (ArUco-scale) |
| press-axis noise | 0.05 rad per episode | randomisation |
| arm base height | +-3 cm per episode | randomisation (only measured transfer failure mode, PL 6.22) |
| joint velocity limits | 1.39 / 1.22 rad/s, proportional scaling | fidelity (Gen3 manual table 46) |
| tool twist caps | 0.5 m/s, 0.8727 rad/s | fidelity (Cartesian joystick mode) |
| joint acceleration limits | 5.2 / 10.0 rad/s^2, proportional scaling | fidelity (table 48, hard) |
| acceleration-limit multiplier | [0.6, 1.4] per episode (from e6ab on, inherited by `arm_sb`/`arm_d`) | randomisation (variant B, PL 6.46) |
| loop dead time | 0-2 control steps, drawn per episode | randomisation from measurement |
| E-stop profile | 20 N / 6 mm latching | fidelity |

Limits are enforced in a hook after every physics substep (clip/scale velocity, then
q = q_prev + v dt), verified on the physical effect, not on the parameter (PL 6.32, 6.37).

### 3.7 Environment versions (needed to read any number)

| id | date | what defines it |
|---|---|---|
| E-A | July | 7DOF, original (wrong) geometry: base effectively 1.56 m, board 1.24 m, no roll, no wall |
| E-B | 08-02/05 | 6DOF, audited geometry (board 0.60, base 0.65, roll 45), physical wall from 08-05 |
| E-C | 09-01 | Webots-measured geometry (0.64 / 0.62 / r 0.107) + perception-level noise |
| E-D | 09-15 | + Gen3 manual velocity limits (scale) and tool caps, E-stop 20 N / 6 mm |
| E-E | 10-02/04 | + Spot body and camera, stand-off -0.34, new start pose |
| E-F | 10-05/06 | + acceleration limits 5.2/10 in per-joint `clip` |
| E-G | 10-07/09 | acceleration limits in `scale` semantics (as the firmware); training keeps the per-episode limit multiplier [0.6, 1.4] from e6ab |
| E-H | 10-09/10 | + training with dead time [0, 2] steps; evaluated at d = 0/1/2 |

## 4. Training methodology

- PPO, lr 3e-4 (fine-tunes 1e-4, final ones 3e-5), n_steps 2048, batch 64, 10 epochs,
  gamma 0.99, lambda 0.95, clip 0.2, 8 envs; `target_kl` 0.03 and `log_std_max` 0.0 from
  September on.
- Curriculum that worked: scratch on loose conditions (free gripper, no finger gate,
  collision cost 0.5, depth shaping) -> fine-tunes that tighten towards the hardware
  protocol, one variable at a time. Scratch directly on the full v2.3 reward collapses to
  "do nothing" (0/20 7DOF, 0/100 6DOF `sc2_full`).
- Evaluation: `training/compare_models.py`, paired episodes (np.random re-seeded with
  seed + k before episode k), so every model sees identical boards, orders and noise
  sequences. Paired sign/binomial tests on discordant pairs; Wilson 95 % CI for n = 100.
  Fisher's exact test (printed by the tool) is too conservative for paired designs
  (0.079 vs 0.00098 on the same data, PL 6.23).
- Pre-registered criteria were written into configs before each run and judged literally.

## 5. Results timeline (main line only)

| stage | model | env, h | result | PL |
|---|---|---|---|---|
| 7DOF reward v2, scratch | v4 @6M | E-A, h1000 | 89/100 | 6.3 |
| 7DOF v2.1 (collision 5.0, action rate) | @8M | E-A, h1000 | 100/100, collisions 15.8 -> 2.4 steps/ep | 6.3 |
| 7DOF v2.3 (finger gate) | @10-12M | E-A, h1000 | 85-90/100 | 6.3 |
| 7DOF + supervisor with retry | v2.3 | E-A, 75 s | 92/100 | 5.2 |
| 6DOF v4-like on fixed geometry | 6M | E-B, h1000 | 0/50 (depth stalls at 7.6 mm) | 6.16 |
| + depth shaping (one variable) | depress_ft 14.23M | E-B, h1000, free gripper, no wall | 80/100 | 6.17 |
| hardware protocol (closed gripper, gate, v2.3) | s2 16.43M | E-B + wall | 51 / 74 / 85 at h 1000/1500/2000 | 6.19-6.23 |
| Webots geometry + noise | s2 | E-C, h1500 | 53/100 (before) | 6.24 |
| log_std clamp, scratch -> fine-tune | ft_sc1_v23_long 15.91M | E-C, h1500 | 76/100, collisions 37.2 -> 7.9 | 6.25 |
| same model under manual limits + E-stop | model A | E-D, h2000 | 1/20 | 6.37 |
| fine-tune in hardware envelope, lr 1e-4 | 17.81M | E-D, h2000 | 7/40, then degrades to 0/40 | 6.38 |
| same, lr 3e-5, +6M, +6M | 28.02M | E-D, h2000 | **92/100**, 23.9 s, 2.3 collision steps | 6.39-6.40 |
| 28.02M in the Spot scene, unchanged | 28.02M | E-E, h2000 | 20/40, 446 Spot-contact steps/ep | 6.42 |
| new stand-off + start pose, fine-tune | spot_e2b 50.22M | E-E, h2000 | 100/100, 7.4 s | 6.42 |
| + acceleration limits (clip) | e5 74.12M | E-F, h2000 | 38/40, 17.5 s | 6.44-6.45 |
| | e6ab 86.12M | E-F, h2000 | 100/100 (clip); 99/100, collisions 14.65 in `scale` | 6.50-6.51 |
| train in `scale`, acc penalties removed | arm_sb 90.12M | E-G, h2000 | **100/100**, 12.8 s, collisions 6.98 (median 4) | 6.51 |
| + dead-time randomisation [0,2] | **arm_d 98.12M** | E-H, h2000 | 98 / 99 / 98 % at d = 0/1/2 | 6.55 |

**Hardware model: `arm_d`** (`models/arm_d_98123552.zip`).
Compared with `arm_sb` on identical seeds: lower collisions at every delay
(3.46 vs 6.98 at d=0, p = 0.0002; 5.62 vs 10.15 at d=1, p = 0.041; 18.74 vs 30.68 at d=2,
p < 0.0001), 98 vs 93/100 at d = 2, at the cost of 2 lost episodes at d = 0 (98 vs 100/100,
p = 0.50). Revert to `arm_sb` only if hardware shows dead time persistently < 50 ms without a
tail. Contract of the scene per checkpoint: `models/README.md`, `training/configs/arm_d.yaml`.

## 6. Key findings (what the thesis can argue)

### 6.1 Penalties added to a working policy mostly failed; structural changes worked
Six attempts to fix behaviour through reward failed: v2.2 finger-misuse penalty (0/100, the
policy learned to avoid buttons), v2.4 contact-quality penalties (85 -> 70 -> 43 %), v2.5
corrected penalties (83 -> 61 %), s3 reward clipping (no effect, more collisions), approach
speed penalty (no effect), acceleration-demand penalty (measured controller ripple, PL 6.48).
The v2.4 collapse had a measurement artefact at its root: the target advances on the press
step, so the lingering contact with the just-pressed cap was taxed as "wrong button"
(~93 % of all "wrong contact" steps). The one success was `collision_cost` 0.5 -> 5.0 for
arm/board contact (collisions 4.7x lower). What did work were structural changes: geometry,
horizon, a log_std clamp, the right limit semantics, the stand-off.
This is an empirical observation for penalties added during fine-tuning, not a general rule:
other reward designs (e.g. an action-rate weight ramped up over training, as in standard Isaac
Lab tasks) were not tested and remain a legitimate option, e.g. for jitter at 8 Hz.

### 6.2 The horizon was hiding most of the "plateau"
`s2` at h = 1000 / 1500 / 2000: 51 / 74 / 85/100, paired 23:0 and 11:0. The max success
time always sat just under the limit (959/1000, 1429/1500, 1926/2000): for a truncated
distribution the observed maximum always looks safe. The same reasoning error was made twice
in a row (PL 6.21, 6.23).

### 6.3 Geometry audit: four defects invalidated a month of 6DOF results (PL 6.15)
Forced-closed gripper from step 0 (costs 32 pp on a trained 7DOF policy, A/B 33/40 vs
20/40), arm base 0.91 m too high (`set_base_xpos` sets the pedestal bottom), board at 1.20 m
instead of 0.60 m, missing 45-degree roll. The "hard button btn3" and "6DOF manipulability"
explanations were wrong; btn3 was simply the farthest button in a wrong scene (7DOF hits it
95/100 under the same code). After the fixes, the 7DOF v2.3 model scored 0/100 on the new
geometry; ablation: relative target encoding absorbs the board roll (D: 2.04 presses/ep), but
moving the base is fatal (C: 0.12), because joint angles and `eef_pos` are absolute.
**Consequence for transfer: the arm mounting height must match the simulation.**

### 6.4 log_std explosion made the policy bang-bang (PL 6.25)
Entropy bonus plus SB3's pre-env action clipping gave a free reward channel: sigma grew to 21,
|mu| to 14.8, 96 % of action components saturated, so the policy was a switch, not a
regulator. A callback capping log_std at 0 (sigma <= 1) gave, at matched budget (~16M):
53 -> 76/100 (paired 34:11, p = 8e-4), collisions 37.2 -> 7.9 steps/ep, smoothness 2.4x.
A first sweep wrongly "refuted" this hypothesis by comparing 4-6M-step runs with a 16M-step
baseline; the run evaluated at 7.59M as "flat" broke through 0.5M steps later.

### 6.5 Velocity limits: the right number and the right semantics (PL 6.31-6.40)
- Applying 0.8727 rad/s per joint (from `joint_limits.yaml`) dropped the best model to 0/40,
  and three curricula, a fine-tune and a scratch could not recover it.
- The Gen3 manual (R07) showed that 0.8727 is the admittance/force-control limit; Cartesian
  joystick (twist) mode uses 1.39 / 1.22 rad/s with **proportional scaling of all joints**
  (direction preserved) plus tool caps 0.5 m/s / 0.8727 rad/s.
- The first limit implementation leaked (wrist up to 2.9x the limit, saturating qvel before
  the step), and an OOM came from a reference cycle in the limit hook (old MjSim freed only by
  cyclic GC). Both found by testing physical effect and memory over many hard resets.
- With the correct envelope, a fine-tune at lr 1e-4 peaked at 25 % and degraded; the same at
  lr 3e-5 rose steadily to 92/100 (std decreasing instead of saturating at the clamp).

### 6.6 Acceleration limits and a measurement-scale error (PL 6.44-6.49)
- Table 48 hard limits 5.2 / 10 rad/s^2: the 100/100-in-5.5 s model relied on 15-27 rad/s^2
  peaks; the honest hardware number became 39/40 in 18.8 s.
- "Demand above the limit" (62 -> 78 % of steps) was chased for three stages. It was not a
  pathology: with the limiter in the loop, demand above it neither helps nor costs, so its
  gradient is zero (the clipped-action problem of Fujita & Maeda, ICML 2018).
- The penalty and the saturation sensor were computed per 2 ms substep, where the OSC ripple
  of low-inertia wrist joints reads hundreds of rad/s^2 even with zero action (MuJoCo's
  qacc is (v+ - v)/h and depends on h). At the 50 ms control scale typical motion fits the
  envelope; only peaks violate it.
- robosuite's `LinearInterpolator` cannot be added after training (0/8); smoothing filters
  (EMA) and command-rate limiters also destroy a trained closed-loop policy. **Envelope and
  smoothness must be inside training, never in the execution path.**

### 6.7 Clip vs scale semantics and control arms (PL 6.50-6.51)
Our per-joint `clip` for accelerations flattered the policy: in `scale` (firmware semantics)
collisions doubled (7.02 -> 14.65, p = 0.035, effect in the tail). Four parallel cloud arms
from one checkpoint, +4M each, one change per arm:

| arm | change | collisions mean (median) | time |
|---|---|---|---|
| K | none (control) | 14.91 (8.0), vs e6ab p = 0.55 | faster, p = 0.022 |
| S | train in `scale` | 6.59 (4.0), vs K p = 0.0001 | same as K |
| SM | `scale` + tighter 4.0/8.0 limits | 15.13 (3.0) | 21.7 s, slower |
| SB | `scale`, acceleration penalties removed | 6.98 (4.0), 100/100 | 12.8 s |

The control arm separated two effects: collisions improve because of the semantics, speed
because of the extra steps. Without arm K the improvement of S would have been unattributable,
which is exactly what happened in an earlier stage (PL 6.46).

### 6.8 Scene geometry beats penalties (PL 6.42)
Adding Spot's body to the scene dropped the 92/100 model to 20/40 with 446 contact steps/ep.
Two stages of Spot-contact penalty (up to 5.0) reduced contact 5.9x at a loss of success.
Moving the stand-off by 24 cm gave contact 0.0 and collisions 0.2 immediately. A symmetric
start pose (j1 = 0) fixed a 0/40 mirror button caused by an optimiser bound at j1 = -0.4.

### 6.9 Dead time (PL 6.52-6.55)
- robosuite has zero loop delay. `arm_sb` without delay training: 40/40 at 1 step (50 ms),
  37/40 at 2 steps, 30/40 at 3 steps; the cliff is between 50 and 100 ms.
- Webots measurement: median 24 ms, range 0-80 ms (sim clock). Kinova documents 40 Hz
  high-level commands and feedback (25 ms quantisation each way).
- Training with delay drawn per episode from [0, 2] steps (range from measurement, unlike the
  failed `gain_rnd` [0.3, 1.2] guess that raised collisions 9.4x) improved collisions and time
  at every delay without hurting success. The failure mode at d = 2 is the same for both
  models (horizon exhausted, contact with Spot's body) but rarer and lighter after training.
- Not modelled: perception delay; no action history in the observation (Katsikopoulos &
  Engelbrecht 2003, DiAReL), so the delayed problem is not Markov for the policy.

### 6.10 Press detection on hardware
Force + TCP kinematics cannot detect presses for these policies: they push at > 20 N for tens to
hundreds of steps before activation, with the TCP up to 10 cm off the button axis (7DOF v2.3,
PL 5.2). For `arm_d` a geometric grip-point criterion catches at best 32/150 activations: at
activation the gripper is at a median 49 degrees to the press axis with 30 mm lateral offset
(it presses with the side of the fingers, PL 6.61). First hardware sessions therefore use the
operator as the detector (the E-stop latch is visible); a "virtual button" (gripper geometry
penetration into the cap, in the bridge's MuJoCo model) is the next automatic candidate.

## 7. Sim2sim in Webots

What Webots verified: kinematics (core FK vs Webots TF 0.2 mm, 0.00 degrees), the observation
vector (blocks equal to 0.17 mm; actions equal to 0.001), stand-off reproduction, ArUco
perception (1.1 mm, 0.88 degrees), and the direction of the first approach.

What Webots cannot measure: policy success. Buttons have no physics, the arm passes through
the board, the gripper is a 2F-140 instead of the 2F-85, and the execution path realised only
~13-14 % of commanded motion through MoveIt Servo or a joint trajectory controller. Root cause
found late (PL 6.60): the Gen3 proto motors had no `controlPID`, so Webots used P = 10
(tau = 0.1 s); with P = 40 realisation rose to 1.00 (translation) and 0.80 (rotation). Earlier
"ceilings" were also partly caused by trajectory horizons shorter than the control period
(goal preempted every step) and by command flooding (each new trajectory replaces the old).
Even with these fixes the closed loop does not complete the task (0/5 on a sequence).

## 8. Hardware deployment pipeline (`ros2_bridge/`)

```
Kortex /joint_states (wrapped to [-pi, pi])
  -> JointUnwrapper (continuous j1/j4/j6 branch anchored at init_qpos)
  -> ObsCore: same angles in the training MuJoCo model -> exact [120] observation
     board pose from ArUco (or nominal), gripper block = training medians
  -> PPO arm_d
  -> ActionToTwist: robosuite filter v+ = alpha v + beta a, caps 0.5 m/s / 0.8727 rad/s
  -> base -> TOOL frame, grip point -> Web App TCP (v += w x r)
  -> /twist_controller/commands (firmware: proportional joint scaling)
```

- **FK from MuJoCo, not TF.** Hand-built conventions (grip offset 0.1448 m, flange rotation,
  quaternion sign rule) broke on the `arm_d` pose; the core reuses the training model and
  reads robosuite observations directly. Offline verification: proprioception bit-exact;
  closed loop with median gripper values 96/100 vs 98/100 with true values (n = 100,
  McNemar p = 0.69, pre-registered fine-tune criterion not met).
- **Startup check.** Core FK vs robot TF must agree within 5 mm / 2 degrees or the bridge
  refuses to run (wrong joint zeros, directions or frames).
- **Unwrapping is critical.** In `arm_d` episodes j6 is always in [3.14, 6.37] and j4 exceeds
  pi in 71 % of steps; `ros2_kortex` wraps to [-pi, pi].
- **Controllers.** JTC for homing (collision-checked path in the training model; straight
  joint interpolation self-collided in Webots), `twist_controller` for the policy; automatic
  switching with zero twist before every handover.
- **Step test** (`--step-test`): +-A twist pulses per axis, first-order-plus-dead-time fit
  (L and tau from 10 %/63 % times, K from the pulse tail); the bridge filter is retuned so that
  (bridge filter -> arm) matches robosuite, otherwise the arm's own lag counts twice.
  Recovered L, tau, K within a few ms on a mock robot.
- **Safety.** Internal watchdog (zeros after 0.1 s without a command); independent deadman
  process on a heartbeat (`twist_controller` holds the last command: on a mock, killing the
  bridge left the arm moving 0.9 rad in 2 s; with the deadman, 0.0 rad after a 154 ms trip).
  Virtual wall for the in-air test (braking v <= gap / t_brake); with a real board it failed
  its pre-registered robosuite criterion three times (39 -> 33-36/40 with STOPs), so it is used
  only in the air.
- **Perception.** ArUco DICT_4X4_50 id 0, 48 mm, averaged over 20 frames on a still arm
  (board pose constant per episode, as in training). Planar-marker PnP ambiguity resolved with
  IPPE + iterative candidates, a "board faces the robot" prior, then min reprojection error
  (IPPE alone picked a 28-degree-wrong solution on noise-free synthetic data).
- **Operator mode** (`--operator`): Enter = pressed, n = skipped, q = stop; logged per step.
- **Stand-off tolerance** (robosuite, `arm_d`, n = 40): from 2 cm farther to 4 cm closer than
  trained; farther loses reach, closer raises collisions (34.9 steps/ep at 6 cm). Yaw tolerance
  not measured.
- **Mock `ros2_kortex`** (`tools/mock_kortex.py`): tests the bridge logic without the robot, not
  the policy's success.
- **Code structure (PL 6.64).** `core/` (no ROS: observation, filter, step-test analysis, wall,
  operator, ArUco), `robots/` (one backend per file over a shared base: joint states, TF, JTC
  homing, watchdog; a backend adds only its output and start/stop hooks), `bridge_node.py`
  (CLI, startup checks, loop), `tools/`. Regression on the mock: step test and a 120-step policy
  run match the pre-split code (grip-to-target distance within 1.5 mm, timing jitter).
- **Upstream driver with mock hardware** (`kortex_bringup`, `use_fake_hardware:=true`, PL 6.64):
  core FK vs the Kortex URDF (6DOF + 2F-85) 0.2 mm / 0.00 deg; switching to `twist_controller`
  is rejected by mock hardware, as documented upstream.

### 8.0 Arm PS: one policy across Spot postures (PL 6.76)

P8F +16M with the arm base height drawn per episode from [0.20, 0.65] m, stand-off keeping the
base-to-board distance, EEF observed relative to the base. n = 100, d = 0: base height 0.62 m
100/100 (26.9 s, 1.0 contact steps/ep), 0.40 m 100/100 (24.2 s, 1.1), 0.20 m (lying) 92/100
(42.5 s, 19.6 contact steps/ep, Spot contact 4/100); d = 1: 0.62 m 100/100, 0.20 m 99/100
(13.3). Pre-registered criterion failed in 1 of 12 checks (board contacts at 0.20 m). P8F
zero-shot at 0.20 m was 2/20 with 213 contact steps/ep.

### 8.1 Lab-driver backend for the 8 Hz policies (PL 6.77)

```
lab /joint_states (j1/j4/j6 in [0, 2pi), velocities present)
  -> JointUnwrapper -> ObsCore built from the TRAINING CONFIG (noise off, nominal base;
     obs_base_relative needs no shift, the board pose in the base frame carries the posture)
  -> PPO P8F/PS
  -> RefPlant: training model set to the measured (q, qd), action through the training chain
     (in-loop EMA 0.5, OSC, twist/joint caps, acceleration limits) for one 124 ms period,
     contacts off -> mean grip-point twist
  -> TOOL frame from FK, grip point (0.2075 m) -> Web App TCP (0.20 m), x2 (driver halves)
  -> /twist_controller/commands (TwistStamped); start pose via saved actions
     nuc -> Home -> rl_via -> rl_start (/goto_position)
```

- **Driver choice.** The lab's `kortex_ros2_python` instead of `ros2_kortex`: the latter switches
  the arm to LOW_LEVEL and cannot run beside the lab stack, needs `ros2_control` from apt on the
  robot PC, and the lab deploys its own driver. Measured on 18.11: dead time 30-80 ms, tau
  ~85 ms, smooth at 8 Hz, no stops over Wi-Fi, Cartesian Joystick caps (0.2 m/s) apply to API
  twists.
- **Action -> twist is not a per-axis gain.** robosuite's OSC couples rotation and translation:
  a rotation command moves the grip point (v_y <- e_wz 0.35, v_z <- e_wy -0.21 m/s per unit).
  Per-axis gains reproduced the simulated linear direction with median cosine 0.68 and failed
  in closed loop (0/5 targets on a kinematic mock); a 6x6 matrix gave cosine 0.95 but stayed
  order-sensitive. Using the training plant itself from the measured state (RefPlant) matches
  simulated free-space velocity with a median error of 3 mm/s at ~30 ms per step. No extra
  first-order filter: the arm's own lag (60 ms + 85 ms) replaces robosuite's (50-170 ms).
- **Driver dead band irrelevant.** 0 of 2298 P8F commands fall below it; near-target grip
  speed median 0.16 m/s.
- **Verification.** Core observations equal the env's bit for bit across random postures
  (0.34-0.47 m base height), up to the 6 mm cap depression in `target_rel`.
- **Physics mock** (`tools/mock_labpy.py`: training scene with board and caps, joint servo
  following the twist with the measured lag, own /clock): 3/5 targets for P8F and PS vs
  99-100/100 in simulation. Large tool-to-axis angles also occur in training (median 14-83 deg
  for btn0-4): the policy presses side caps with the side of the long 2F-140 fingers, presses
  register with the grip point 2-8 cm from the cap centre, so target switching must come from
  press detection, not distance. The remaining gap is contact behaviour (stiff servo vs
  compliant OSC); only hardware can settle it.
- **Route hazard at the lying posture.** With the board at its current height and the trained
  stand-off (0.82 m at 0.20 m base height), the rl_via -> rl_start sweep crosses the centre cap
  (-1.9 cm); rl_start itself is clear. Procedure: go to rl_start while standing at 0.93 m
  (>= 11 cm from the board), approach to 0.82 m (>= 5.5 cm), lie down (clearance grows to
  19.5 cm). Lying down in place at 0.93 m puts the board centre 1.03 m from the base, outside
  PS's training distribution (0.93 m).

## 9. Negative results worth reporting

| hypothesis / intervention | outcome | PL |
|---|---|---|
| reward penalties for contact quality (v2.2, v2.4, v2.5) | all collapsed or degraded | 6.3-6.6 |
| reward clipping prevents late collapse (s3) | refuted, collisions up | 6.20 |
| gain randomisation [0.3, 1.2] for mechanism robustness | collisions 9.4x up, no gain | 6.30 |
| curricula on joint velocity limit (3 variants) | no adaptation at 0.8727 | 6.32, 6.34 |
| smaller OSC output_max (rotation 0.1, translation 0.02) | rotation tunes, translation breaks the task | 6.35-6.36 |
| approach-speed penalty | no measurable effect | 6.45 |
| acceleration-demand penalty / saturation sensor | measured substep ripple, not behaviour | 6.46-6.48 |
| tighter-than-hardware envelope in training (SM) | median better, mean and time worse | 6.51 |
| post-hoc EMA filter, command-rate limiter, OSC interpolator | destroy a trained closed-loop policy | 6.42, 6.46, 6.49 |
| press detection from force + TCP kinematics / grip geometry | infeasible (32/150 at best) | 5.2, 6.61 |
| virtual wall with a real board | fails pre-registered criterion 3x | 6.62 |
| Webots closed loop as a success benchmark | cannot measure success | 6.58-6.60 |

## 10. Methodological lessons (protocol rules adopted)

1. Report horizon **and** environment version with every success rate.
2. A truncated distribution's maximum always looks safe; only a longer horizon decides.
3. Pass a known-good model through any new envelope (limits, geometry, mechanism) before
   designing a multi-hour run; also check the curriculum's start point, not only its end.
4. Verify every constraint mechanism on its physical effect and every sim hook on memory over
   many hard resets, not on the parameter value.
5. Pre-register the criterion and the stop rule in steps; do not move the goalposts on a
   flat curve (sparse rewards with threshold shaping break through late).
6. A borderline result at n = 20 does not justify a multi-hour decision; confirm on new seeds
   (n = 20: p = 0.057 -> n = 40: p = 0.20; best at n = 40 was worst at n = 100).
7. Use paired tests on discordant pairs, not Fisher, for seed-paired designs.
8. Heavy-tailed metrics (collisions): compare medians and paired tests, aggregate per episode
   first, never one median over pooled steps.
9. `best_model` selected by eval reward is misleading when reward is dominated by shaping or
   penalties; select by success on paired evaluation.
10. A parameter duplicated in two places changes silently in one (hard-coded noise in tools,
    missing `base_x_offset` in the evaluator, missing flags in the recorder). One source of
    truth; tools take defaults from the env.
11. A scripted test's failure does not prove infeasibility, only that the strategy is bad
    (12 mm "impossible" for 6DOF was a scripting artefact).
12. Attribute improvements only with a control arm (same extra steps, no change).
13. Shutdown and supervision logic must live on the machine, not in the supervising session.

## 11. Limitations and open issues

- No hardware result yet. Unmeasured on hardware: wall-clock loop dead time and its tail,
  perception delay, arm step response (filter retuning), TCP offset in the Kinova Web App,
  camera extrinsic (Kinova Vision calibration), camera topics, marker placement on the board.
- **Gripper mismatch (found 2026-10-29):** the lab arm carries a Robotiq 2F-140 (Web App;
  TCP Z 0.2 m, 1.025 kg), while `arm_d` was trained with robosuite's `Robotiq85Gripper`.
  The 2F-140 grip site lies 62 mm farther from the flange (0.2075 vs 0.145 m); unadapted, the
  policy would drive the real fingers about 6 cm into the board. The observation size is
  unchanged (6 gripper joints), so `arm_d` can be fine-tuned in a 2F-140 scene. The MASKOR
  Webots model already uses the 2F-140.
- Hardware scene (docs 6.66): 2F-140 plus the lab E-stop boxes (cap top 80 mm above the face
  vs 53 mm). `arm_d` there, n=20, h=2000: still 20/20, but board-contact steps per episode
  2.0 -> 35.3 and Spot contact 0/20 -> 8/20. The boxes alone change nothing (1.4); the gripper
  alone gives 103.8. Fine-tuning in the hardware scene is required, with collisions as the
  criterion.
- Arm G (docs 6.68): `arm_d` +8M in the hardware scene with training speed caps. n=100, h=2000:
  100/100 at d=0 and d=2, board-contact steps 2.6 / 8.9 per episode (vs `arm_d` in its own
  scene 3.5 / 18.7), but Spot contact in 27 / 52 episodes (vs 2 / 11), so it fails the
  pre-registered criterion on Spot contact. Arm L (lab caps) was preempted at 99.7M and resumed.
  Diagnosis (n=30): the wrist links touch the front camera bar / front edge of Spot, mostly at
  the centre and lower-bevel buttons; the 2F-140 (+62 mm) and the taller caps (+27 mm) pull the
  wrist ~9 cm back over Spot at the 2F-85 stand-off (-0.34 m). Parking Spot 6-9 cm farther
  (-0.40 / -0.43) gives 0/30 Spot contact at d=0 and 3-4/30 at d=2 without retraining, at the
  cost of longer episodes (n=100 confirmation pending).
  Stand-off sweep with clearance (n=30, -0.34...-0.52): the usable window is -0.40...-0.43;
  beyond -0.46 reach fails (24/30 -> 1/30). At -0.43, d=0: 30/30, no Spot contact, minimum
  clearance 3.5 cm; d=2: 30/30 but 3/30 contact (min -1.6 cm), so no stand-off removes contact
  under dead time. The start pose itself clears the camera bar by only 0.7-0.9 cm.
- Arm L (docs 6.68, lab Web App caps 0.2 m/s, 20 deg/s, joints 50 deg/s): 0/17 at d=0 and d=2
  (h=2000, evaluation stopped once the >= 90/100 criterion was unreachable); every episode hits
  the horizon with 1.6-1.9 of 5 buttons pressed. 8M steps of fine-tuning from `arm_d` do not
  adapt the policy to the lab speed limit.
- Arm L2 (docs 6.70): arm L continued for +16M with horizon 4000 (lab caps). A longer horizon
  should not by itself teach slowness: time pressure comes from discounting (gamma 0.99, ~100-step
  effective horizon), and the only per-step presence term (<= 0.05/step) sums to <= 5 points,
  below one press (25). Criterion: >= 90/100 at h=4000 (n=100, d=0 and d=2).
- Slower control loop under the lab limits (docs 6.71): the lab caps are exactly 0.4 x the
  training caps (linear and angular), so at ~8 Hz the cap per control step matches the 20 Hz
  training loop; velocity observations are scaled by 20/f. A robosuite pitfall: with a
  non-integer number of physics substeps per control step (8 Hz = 62.5 x 2 ms) observations are
  sampled one control step late; 500/62 = 8.065 Hz is used and the env now rejects such rates.
  Arm P8 fine-tunes P for +8M at 8.065 Hz with the lab caps and delay [0, 1] step. Criterion:
  >= 90/100, Spot contact <= 5/100, board contact <= 2.0 steps/ep, time to success <= 800 steps.
- Arm P8 (n=100, lab caps, 8.065 Hz, -0.43): 99/100 (d=0) and 100/100 (d=1) in 36-43 s, Spot
  contact 3-4, board contact 1.2 / 3.7 steps/ep (fails <= 2.0 at d=1). Jitter (docs 6.72, 500 Hz):
  bang-bang actions, TCP direction reversals 1.10/s vs 0.16/s for P at 20 Hz; a deployment-only
  action EMA trades jitter for collisions. Arm P8F (docs 6.73) trains with the EMA (alpha 0.5)
  inside the loop; criterion adds median reversals <= 0.3/s and tremor <= 1.0 mm.
- Arm P8F result (docs 6.73, n=100, h=2000 at 8.065 Hz, lab caps, -0.43, in-loop EMA 0.5, lab camera
  box): 99/100 (d=0) and 100/100 (d=1) in 26-29 s, Spot contact 0/100 at both, board contact 1.0 / 3.0
  steps/ep (fails <= 2.0 at d=1); jitter n=20: median TCP reversals 0.41/s (d=0, fails <= 0.3) and
  0.25/s (d=1), tremor 0.94 / 0.83 mm. Criterion missed in 2 of 12 checks, but better than P8 on every
  metric (reversals 1.10 -> 0.41/s, tremor 1.44 -> 0.94 mm, time 36 -> 26 s, Spot contact 3-4 -> 0):
  an action filter inside the training loop reduces jitter without the collision cost of a
  deployment-only filter. P8F becomes the hardware model.
- Arm PS (docs 6.76): one policy for lying / crouched / standing Spot. Arm base height drawn per
  episode from [0.20, 0.65] m (lying measured 0.20 m), stand-off keeps the base-board distance,
  EEF position observed in the arm base frame. P8F zero-shot (n=20): 20/20 at 0.62 and 0.50 m,
  15/20 at 0.40, 7/20 at 0.30, 2/20 at 0.20 (212 board-contact steps/ep). +16M from P8F 130.12M;
  criterion: >= 90/100 at 0.20/0.40/0.62 m and >= 95/100 at 0.62, Spot contact <= 5/100,
  board contact <= 2.0 steps/ep.
- Measured lab front camera (docs 6.74): ifm O3R ToF head, 7.5-10 cm in front of the arm base
  axis, top 4.5 cm above Spot - closer and higher than the 28 cm Kinect bar used in training.
  Added as an extra collision box with a 1.5 cm margin (`lab_cam_box`). n=50, lab caps, -0.43:
  P8F at 128.42M has 0/50 Spot contact at d=0 and d=1 and never comes closer to the camera box
  than in the start pose (9.7 cm); the Kinect bar stays the binding buffer (min 2.5 cm).
  P8 entered the margin box once (-1.8 cm). No scene change needed for training.
- Lab session 2026-11-12 (docs 6.75, read-only): `/joint_states` at ~1 kHz with gaps up to 0.56 s
  (bridge needs an observation-age guard); the lab driver sends each twist twice, stops after
  0.20-0.25 s without a message, and also offers a 1 kHz low-level `FollowJointTrajectory` path
  (hardware route for a position-action policy, future work). Saved robot poses were checked in
  simulation: the Web App angle convention matches the sim; rest/nuc -> RL start is clear, but a
  straight joint move Home -> RL start dips the gripper 0.5-2.6 cm into Spot's front, so a via
  point is used. Home as the training start pose was considered and deferred (needs fine-tuning,
  off-distribution wrist branch); it returns if the measured start clearance is < 2 cm.
  The lab already runs a classical visual-servoing E-stop pipeline - a baseline for comparison.
- Arm L2 was stopped at 118.02M: the 4M checkpoint rule failed (eval episode length 4000).
- Arm P (docs 6.69): `arm_g` +8M at the chosen stand-off (-0.43, i.e. arm base 0.93 m from the
  wall) with a new per-episode stand-off randomisation of +/-2 cm (arm and Spot move together).
  Pre-registered criterion (n=100, h=2000): success >= 95/100, Spot contact <= 2/100 at d=0 and
  <= 5/100 at d=2; at -0.41 and -0.45 (d=0) success >= 95/100 and Spot contact <= 2/100.
  Result (n=100, h=2000): 100/100 at -0.41, -0.43 (d=0 and d=2), 99/100 at -0.45; Spot contact
  2 / 0 / 7 / 0 episodes; board-contact steps 0.4-1.5 per episode; time to success 192-258 steps.
  Fails the criterion only on Spot contact at d=2 (7 > 5; arm_g: 10).
- Lab Web App speed limits (Cartesian Joystick 0.2 m/s, 20 deg/s; docs 6.67), hardware scene,
  n=20, h=2000: `arm_d` 20/20 with training caps -> 2/20 with the lab twist caps, 1/20 adding the
  50 deg/s joint limit. The linear cap alone gives 0/20; the angular cap alone 16/20. The policy
  gets lost rather than slowing down. Either the lab raises the limit or training uses it.
- One controller snapshot disagrees with the core's nominal FK by about 8 mm / 0.6 deg
  (vs 0.2 mm against the Kortex URDF); more poses are needed.
- Web App joint limits in Angular Joystick mode: 50 deg/s, 1 rad/s^2 (joints 1-3) and
  10 rad/s^2 (4-6), vs 1.39/1.22 rad/s and 5.2/10 rad/s^2 in training; twist-mode limits unread.
- Gripper observation is a constant (training medians).
- Button model from a datasheet, not from the lab's board; force profile and stroke unmeasured.
- Board dimensions taken from the MASKOR Webots model (one socket inconsistent), not measured
  with a tape on the real board.
- Stand-off yaw tolerance unmeasured; mounting height must match within the +-3 cm trained.
- No action history in the observation under dead time.
- Contact in MuJoCo and Webots/ODE is modelled differently and cannot be tuned to agree.
- Retry in the old supervisor retreats along the press axis and reproduces the same joint-4
  limit configuration that causes most "stuck" episodes (73.6 % of stuck time near the j4
  limit vs 20 % for successes, p = 0.0031); a configuration-changing retry is future work.
- 7DOF and 6DOF results are from different task versions and are not comparable numerically.

## 12. Artefacts index

- Models and scene contracts: `models/README.md`; configs in `training/configs/`.
- Evaluation CSVs: `logs/board_6dof/eval_YYYY-MM-DD/`.
- Videos: `docs/video/` (e.g. `model_arm_d_successes*.mp4` d = 0/1/2 triptych on identical
  seeds; `model_arm_d_d2_robustness.mp4` vs `model_arm_sb_d2_comparison.mp4`, seed 1253, d = 2:
  `arm_d` 5/5 in 523 steps, `arm_sb` 2/5 with 350 steps of contact with Spot;
  `webots_bridge_aruco_btn0_clip.mp4` Webots + ArUco demo). Episode steps in every recording
  match the evaluation CSV step for step.
- Figures: `docs/img/` (learning curves, geometry before/after the audit).

## 13. References

Hardware and software documentation:
- Kinova, *Gen3 Ultra lightweight robot User Guide*, R07 - tables 46-50 (joint speed,
  acceleration, torque limits; Cartesian joystick behaviour, p. 98-100).
  https://www.kinovarobotics.com/uploads/User-Guide-Gen3-R07.pdf
- Kinovarobotics `ros2_kortex` (`hardware_interface.cpp`: TOOL-frame twist, angle wrapping;
  `joint_limits.yaml`, `twist_limits.yaml` not read). https://github.com/Kinovarobotics/ros2_kortex
- Kinovarobotics `ros_kortex/kortex_driver/readme.md`: high-level commands every 25 ms, 40 Hz.
- applied-ai-lab `kortex_hardware`: high-level servoing 40 Hz, low-level 1 kHz.
- MuJoCo documentation, *Computation - Numerical integration* (qacc as (v+ - v)/h) and *XML
  reference* (armature). https://mujoco.readthedocs.io
- robosuite controller documentation (`LinearInterpolator`, `ramp_ratio`).
  https://robosuite.ai/docs/simulation/controller.html
- MASKOR / FH Aachen `webots_ros2_spot`, branch `kinova_gen3`. https://github.com/MASKOR/webots_ros2_spot
- goldengeko `robosuite_training_examples` (6DOF Gen3 MJCF). https://github.com/goldengeko/robosuite_training_examples
- E-stop datasheet (22/30 mm mushroom button): 5 mm stroke, 25 N, latching.

Literature:
- Fujita, Maeda. *Clipped Action Policy Gradient*. ICML 2018. arXiv:1802.07564.
- Aljalbout, Frank, Karl, van der Smagt. *On the Role of the Action Space in Robot
  Manipulation Learning and Sim-to-Real Transfer*. arXiv:2312.03673.
- Katsikopoulos, Engelbrecht. *Markov decision processes with delays and asynchronous cost
  collection*. IEEE TAC 48(4):568-574, 2003.
- Malmir, Josifovski, Klarmann, Knoll. *DiAReL* (state augmented with recent actions).
  arXiv:2306.09010, IEEE TCST.
- *Non-Prehensile Throwing: A Reinforcement Learning Perspective*. alphaxiv 2609.00771.
- *Generating Realistic Arm Movements in Reinforcement Learning: A Quantitative Comparison of
  Reward Terms and Task Requirements*. arXiv:2402.13949.
- Delay randomisation practice (uniform per-episode delay): survey arXiv:2502.13187.

Rejected sources (do not cite): a third-party "13 Hz gRPC ceiling" claim without method;
arXiv 2606.05236 (search snippet not supported by the paper); `ros2_kortex` PR #241 values as
general limits.

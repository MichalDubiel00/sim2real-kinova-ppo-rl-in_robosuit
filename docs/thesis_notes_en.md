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

**Hardware model: `arm_d`** (`logs/board_6dof/arm_d/checkpoints/ppo_board_98123552_steps.zip`).
Compared with `arm_sb` on identical seeds: lower collisions at every delay
(3.46 vs 6.98 at d=0, p = 0.0002; 5.62 vs 10.15 at d=1, p = 0.041; 18.74 vs 30.68 at d=2,
p < 0.0001), 98 vs 93/100 at d = 2, at the cost of 2 lost episodes at d = 0 (98 vs 100/100,
p = 0.50). Revert to `arm_sb` only if hardware shows dead time persistently < 50 ms without a
tail. Contract of the scene per checkpoint: `models/README.md`, `training/configs/arm_d.yaml`.

## 6. Key findings (what the thesis can argue)

### 6.1 Reward cannot fix behaviour; structure can
Six attempts to fix behaviour through reward failed: v2.2 finger-misuse penalty (0/100, the
policy learned to avoid buttons), v2.4 contact-quality penalties (85 -> 70 -> 43 %), v2.5
corrected penalties (83 -> 61 %), s3 reward clipping (no effect, more collisions), approach
speed penalty (no effect), acceleration-demand penalty (measured controller ripple, PL 6.48).
The v2.4 collapse had a measurement artefact at its root: the target advances on the press
step, so the lingering contact with the just-pressed cap was taxed as "wrong button"
(~93 % of all "wrong contact" steps). The one success was `collision_cost` 0.5 -> 5.0 for
arm/board contact (collisions 4.7x lower). What did work were structural changes: geometry,
horizon, a log_std clamp, the right limit semantics, the stand-off.

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
- **Mock `ros2_kortex`** (`mock_kortex.py`): tests the bridge logic without the robot, not the
  policy's success.

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
- Gripper observation is a constant (training medians); the real 2F-85 is stiffer than the
  simulated Robotiq 85, whose finger joints are loose.
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

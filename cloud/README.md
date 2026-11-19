# Cloud campaign: four control arms

Why the cloud: **not for speed, for control runs.** The biggest methodological gap of
stages 6.43-6.47 was the lack of reference arms: every "improvement" also added millions of
training steps, so it could not be attributed (docs 6.46: collisions 20.9 -> 8.6 at p = 0.047,
but the run "same 4M without the penalty" was never made). Four arms started **in parallel**
from the same checkpoint remove that gap. `arm_d` (docs 6.55) was later run the same way.

## Arms

All: **+4M steps** from `spot_e6ab/checkpoints/ppo_board_86123552_steps.zip`
(hardware model at the time, n=100: 100/100, 15.2 s, collisions 7.0, misuse 0.0).
Absolute target: **90 123 552** steps.

| arm | config | change vs e6ab | purpose |
|---|---|---|---|
| **K** | `arm_k.yaml` | **none** | control: what 4M extra steps alone give |
| **S** | `arm_s.yaml` | `joint_acc_limit_mode: scale` | fidelity fix: the firmware scales proportionally (docs 6.50) |
| **SM** | `arm_sm.yaml` | `scale` + limits **4.0/8.0** | compliance by construction: train tighter than hardware so the firmware never clips |
| **SB** | `arm_sb.yaml` | `scale` + `acc_*` penalties = 0 | do the penalties that measure ripple (docs 6.48) hurt? |

**SM is evaluated at 5.2/10.0**, not 4.0/8.0; otherwise it is a different scene (the same
trap as `base_x_offset` in 6.42). Results: docs 6.51.

## Launch

```bash
# 0. once: bucket
gsutil mb -l europe-west3 gs://BUCKET-NAME

# 1. once: code (~14 MB) + start checkpoint (2.3 MB) to GCS
cloud/bundle.sh BUCKET-NAME

# 2. four spot instances, one per arm
for ARM in k s sm sb; do
  gcloud compute instances create kinova-arm-$ARM \
    --zone=europe-west3-a \
    --machine-type=e2-standard-8 \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --boot-disk-size=40GB \
    --image-family=debian-12 --image-project=debian-cloud \
    --scopes=https://www.googleapis.com/auth/devstorage.read_write \
    --metadata=arm=$ARM,bucket=BUCKET-NAME,target-steps=90123552 \
    --metadata-from-file=startup-script=cloud/startup.sh
done
```

`--scopes=devstorage.read_write` is **required**, and the compute service account also needs
a role on the bucket: without them `gsutil` on the VM cannot write and the runner starts but
saves nothing.

## Preemption

A spot instance can be killed at any time. Three layers of protection:

1. a checkpoint every **100 000 steps** (`checkpoint_freq` in the config) bounds the loss,
2. sync to GCS every **5 min** plus a final sync on SIGTERM (`trap` in `run_arm.sh`; GCE gives
   ~30 s, `TimeoutStopSec=25`),
3. `--instance-termination-action=STOP` + `Restart=always`: after the instance restarts,
   `run_arm.sh` reads the **highest checkpoint in GCS**, computes the remaining steps and
   resumes. Nothing has to be remembered between boots. After reaching the target the runner
   shuts the VM down.

A stopped instance has to be started by hand (`gcloud compute instances start kinova-arm-s
--zone=...`) or by a MIG with autohealing. Check with `gcloud compute instances list`.

> ⚠ **Resuming is safe only because every curriculum in these configs is past its end**
> (`action_rate_anneal_origin: 50223552` at > 86M steps => `frac > 1`, the penalty stays at
> its target 0.3). A config with an **active** curriculum needs an explicit `*_anneal_origin`,
> otherwise every resume restarts the curriculum (the trap from docs 6.32 that once broke the
> `hwlim_curr` run).

## Do NOT change

- **`n_envs: 8`.** With 32 cores 32 envs is tempting, but it changes the effective batch and
  the learning dynamics, and **the whole history (86M steps, all paired tests) is at 8**.
  Four arms with 8 envs each are worth more than one with 32, which is the point of the cloud.
- `n_steps`, `batch_size`, `learning_rate`, `log_std_max`, `target_kl`: same reason.
- Recordings (`watch_board.py`) stay local: they need rendering, evaluations do not
  (`use_camera_obs=False`).

## Progress and results

```bash
# progress (highest checkpoint of each arm)
for ARM in k s sm sb; do
  echo -n "arm_$ARM: "
  gsutil ls gs://BUCKET-NAME/arms/arm_$ARM/checkpoints/ 2>/dev/null \
    | sed 's/.*ppo_board_\([0-9]*\)_steps.zip/\1/' | sort -n | tail -1
done

# download checkpoints for local evaluation
for ARM in k s sm sb; do
  mkdir -p logs/board_6dof/arm_$ARM/checkpoints
  gsutil -m rsync -r gs://BUCKET-NAME/arms/arm_$ARM/checkpoints \
    logs/board_6dof/arm_$ARM/checkpoints
done

# TensorBoard curves
bash cloud/tb_pull.sh BUCKET-NAME
```

Evaluate **locally** with the same `compare_models.py`, on the same seeds as e6ab, in the
target `--acc-mode scale`:

```bash
START_POSE="0.0,1.4692,-2.2410,3.1403,-2.0200,3.1416"
python training/compare_models.py "arm_s=logs/board_6dof/arm_s/checkpoints/ppo_board_90123552_steps.zip" \
  --robot Kinova3_6DOF --episodes 100 --seed 1234 --horizon 2000 \
  --kinova-limits --acc-limits --acc-mode scale \
  --estop-config training/configs/ppo_board_6dof_estop_hw.yaml \
  --spot-body --spot-front-cam --base-x-offset=-0.34 --init-qpos="$START_POSE"
```

Reference (e6ab, `--acc-mode scale`, n=100): 99/100, 16.1 s, collisions **14.65** (median 6.5).
With `clip` it was 7.0: **do not mix these two numbers.**

## Cost

Measured on `e2-standard-8` (2026-10-08): **164-180 steps/s**, i.e. about **6.5 h per arm**
and **~26 arm-hours** for the four (about 6.5 h of wall-clock time, since they run in
parallel). The laptop does ~250 steps/s, so the first estimate (18 h) was 45 % too low:
cloud vCPUs are slower than the laptop's cores. Check current spot prices in the console for
the chosen region. No GPUs on purpose: with a 256x256 MLP an accelerator adds nothing, and
MuJoCo stepping is the bottleneck.

## Arms G and L (2026-11-03, docs 6.68)

+8M from `arm_d` (98 123 552), target 106 123 552, hardware scene (2F-140, lab E-stop boxes).
G keeps the training speed caps; L uses the lab Web App caps (0.2 m/s, 0.349 rad/s, joints
0.873 rad/s).

```bash
cloud/bundle.sh BUCKET-NAME logs/board_6dof/arm_d/checkpoints/ppo_board_98123552_steps.zip
for ARM in g l; do
  gcloud compute instances create kinova-arm-$ARM ... \
    --metadata=arm=$ARM,bucket=BUCKET-NAME,target-steps=106123552,start-ckpt=ppo_board_98123552_steps.zip,start-step=98123552 \
    --metadata-from-file=startup-script=cloud/startup.sh
done
```

Evaluate in the arm's own scene: add `--gripper Robotiq140Gripper --button-xml button_lab.xml`,
and for L also `--twist-limits 0.2,0.349 --joint-vel 0.873`.

## Arm P (2026-11-05, docs 6.69)

+8M from `arm_g` (106 123 552), target 114 123 552, hardware scene at the Spot stand-off found
by the sweep in docs 6.68: `base_x_offset -0.43`, `base_x_noise 0.02` (per-episode stand-off,
arm and Spot move together).

```bash
cloud/bundle.sh BUCKET-NAME logs/board_6dof/arm_g/checkpoints/ppo_board_106123552_steps.zip
gcloud compute instances create kinova-arm-p ... \
  --metadata=arm=p,bucket=BUCKET-NAME,target-steps=114123552,start-ckpt=ppo_board_106123552_steps.zip,start-step=106123552 \
  --metadata-from-file=startup-script=cloud/startup.sh
```

Spot VMs with `--instance-termination-action=STOP` do **not** restart after preemption: check
`gcloud compute operations list --filter="targetLink~kinova-arm-"` and restart with
`gcloud compute instances start`; `run_arm.sh` resumes from the highest checkpoint.
Evaluate with `--gripper Robotiq140Gripper --button-xml button_lab.xml --base-x-offset=-0.43`.

## Arm L2 (2026-11-05, docs 6.70)

`arm_l` continued with `horizon: 4000`, +16M from 106 123 552 to 122 123 552. The start
checkpoint is uploaded under its own name, because `start/ppo_board_106123552_steps.zip` is
already arm P's start (`arm_g`):

```bash
cp logs/board_6dof/arm_l/checkpoints/ppo_board_106123552_steps.zip logs/board_6dof/arm_l/arm_l_106123552.zip
cloud/bundle.sh BUCKET-NAME logs/board_6dof/arm_l/arm_l_106123552.zip
gcloud compute instances create kinova-arm-l2 ... \
  --metadata=arm=l2,bucket=BUCKET-NAME,target-steps=122123552,start-ckpt=arm_l_106123552.zip,start-step=106123552 \
  --metadata-from-file=startup-script=cloud/startup.sh
```

Preemption watchdog (local, stops by itself once every listed arm has `KONIEC`):
`setsid nohup ~/.kinova_watch/watch.sh p l2 &`, log in `~/.kinova_watch/watch.log`.

### Switching a running arm from spot to standard (2026-11-05)

P and L2 were preempted three times in two days. Both were switched to standard provisioning
(no preemption; run_arm.sh resumes from the last checkpoint):

```bash
gcloud compute instances stop kinova-arm-$ARM --zone europe-west3-a
gcloud compute instances set-scheduling kinova-arm-$ARM --zone europe-west3-a \
  --provisioning-model=STANDARD --no-preemptible --clear-instance-termination-action \
  --maintenance-policy=MIGRATE --restart-on-failure
gcloud compute instances start kinova-arm-$ARM --zone europe-west3-a
```

`--no-preemptible` is required, otherwise the API rejects STANDARD. The auto shutdown at the
target still stops the VM, so billing ends with the run.

## Arm P8 (2026-11-06, docs 6.71)

`arm_p` +8M (114 123 552 -> 122 123 552) at `control_freq` 500/62 Hz with the lab Web App caps,
standard VM from the start. Each control step is 62 physics substeps instead of 25, so expect
~2.5x fewer steps/s (~30 h).

```bash
cp logs/board_6dof/arm_p/checkpoints/ppo_board_114123552_steps.zip logs/board_6dof/arm_p/arm_p_114123552.zip
cloud/bundle.sh BUCKET-NAME logs/board_6dof/arm_p/arm_p_114123552.zip
gcloud compute instances create kinova-arm-p8 --provisioning-model=STANDARD --maintenance-policy=MIGRATE ... \
  --metadata=arm=p8,bucket=BUCKET-NAME,target-steps=122123552,start-ckpt=arm_p_114123552.zip,start-step=114123552 \
  --metadata-from-file=startup-script=cloud/startup.sh
```

Evaluate with `--control-freq 8.064516129 --dilate-obs --twist-limits 0.2,0.349 --joint-vel 0.873`.

## Arm PS (2026-11-16, docs 6.76)

`arm_p8f` +16M (130 123 552 -> 146 123 552) with the arm base height drawn from [0.20, 0.65] m
(Spot lying .. standing), standard VM.

```bash
cp logs/board_6dof/arm_p8f/checkpoints/ppo_board_130123552_steps.zip logs/board_6dof/arm_p8f/arm_p8f_130123552.zip
cloud/bundle.sh BUCKET-NAME logs/board_6dof/arm_p8f/arm_p8f_130123552.zip
gcloud compute instances create kinova-arm-ps --provisioning-model=STANDARD --maintenance-policy=MIGRATE ... \
  --metadata=arm=ps,bucket=BUCKET-NAME,target-steps=146123552,start-ckpt=arm_p8f_130123552.zip,start-step=130123552 \
  --metadata-from-file=startup-script=cloud/startup.sh
```

Evaluate with the P8F flags plus `--env-action-ema 0.5 --obs-base-relative --standoff-keep-reach
--base-z-range H,H` for each height H.

## Arm PH (2026-11-19, docs 6.78)

`arm_ps` +16M (146 123 552 -> 162 123 552) started from the lab's Home pose, standard VM.

```bash
cp logs/board_6dof/arm_ps/checkpoints/ppo_board_146123552_steps.zip logs/board_6dof/arm_ps/arm_ps_146123552.zip
cloud/bundle.sh BUCKET-NAME logs/board_6dof/arm_ps/arm_ps_146123552.zip
gcloud compute instances create kinova-arm-ph --zone=europe-west3-a --machine-type=e2-standard-8 \
  --provisioning-model=STANDARD --maintenance-policy=MIGRATE --image-family=debian-12 \
  --image-project=debian-cloud --boot-disk-size=40GB --scopes=storage-rw \
  --metadata=arm=ph,bucket=BUCKET-NAME,target-steps=162123552,start-ckpt=arm_ps_146123552.zip,start-step=146123552 \
  --metadata-from-file=startup-script=cloud/startup.sh
```

Evaluate with the PS flags and `--init-qpos=0.0,0.2618,-2.2689,0.0,0.9599,1.5708`.

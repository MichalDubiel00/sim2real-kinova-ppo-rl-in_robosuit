#!/bin/bash
# Record the hardware model arm_d @98.12M (docs 6.55) into docs/video/ (mp4 is gitignored).
# Use the SAME envelope as training/evaluation, or the video shows a different task
# (docs 6.50). Seed = 1234 + episode number from the evaluation CSV.
cd /home/kwadracik/Batchelor/sim2real-kinova-ppo-rl-in_robosuite
V=docs/video
L=logs/board_6dof/stdout
START_POSE="0.0,1.4692,-2.2410,3.1403,-2.0200,3.1416"
CK=logs/board_6dof/arm_d/checkpoints/ppo_board_98123552_steps.zip
COMMON_FLAGS="--model $CK --robot Kinova3_6DOF --horizon 2000 --kinova-limits --acc-limits
     --acc-mode scale --estop-config training/configs/ppo_board_6dof_estop_hw.yaml
     --spot-body --spot-front-cam --base-x-offset=-0.34 --init-qpos=$START_POSE
     --fps 20 --width 1280 --height 720 --camera board_view"

record () {  # $1=file  $2=seeds  $3=delay ("" = none)  $4=label
  local D=""; [ -n "$3" ] && D="--action-delay $3"
  echo "--- $4 -> $V/$1 (seeds $2, delay ${3:-0}) ---"
  timeout 3600 conda run --no-capture-output -n robosuite \
    python training/watch_board.py --record "$V/$1" --seeds "$2" $D $COMMON_FLAGS \
    > "$L/video_${1%.mp4}.log" 2>&1
  echo "rc=$?  size $(ls -lh $V/$1 2>/dev/null | awk '{print $5}')"
  grep -aE "^ *ep[0-9]|success" "$L/video_${1%.mp4}.log" | tail -4
}

# 1) Representative successes at d=0: fastest, median, hardest
record model_arm_d_successes.mp4 1268,1333,1261 "" "successes d=0"

# 2) Why this model: seed 1253 at d=2, arm_sb loses (350 steps on the Spot body), arm_d wins
record model_arm_d_d2_robustness.mp4 1253 2 "dead-time robustness d=2"

# 3) Failure tail: both lost episodes at d=0
record model_arm_d_failures.mp4 1271,1275 "" "failures d=0"
echo RECORDING_DONE

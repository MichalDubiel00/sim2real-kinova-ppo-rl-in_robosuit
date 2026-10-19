#!/bin/bash
# Triptych: the same three episodes (seeds 1268, 1333, 1261) at d=1 and d=2; with
# model_arm_d_successes.mp4 (d=0) only the loop dead time changes. No memory wait loop.
cd /home/kwadracik/Batchelor/sim2real-kinova-ppo-rl-in_robosuite
V=docs/video; L=logs/board_6dof/stdout
START_POSE="0.0,1.4692,-2.2410,3.1403,-2.0200,3.1416"
CK=logs/board_6dof/arm_d/checkpoints/ppo_board_98123552_steps.zip
COMMON_FLAGS="--model $CK --robot Kinova3_6DOF --horizon 2000 --kinova-limits --acc-limits
     --acc-mode scale --estop-config training/configs/ppo_board_6dof_estop_hw.yaml
     --spot-body --spot-front-cam --base-x-offset=-0.34 --init-qpos=$START_POSE
     --fps 20 --width 1280 --height 720 --camera board_view"
for D in 1 2; do
  OUT="model_arm_d_successes_d${D}.mp4"
  echo "--- successes at d=${D} ($((D*50)) ms) -> $V/$OUT ---"
  timeout 3600 conda run --no-capture-output -n robosuite \
    python training/watch_board.py --record "$V/$OUT" --seeds 1268,1333,1261 \
    --action-delay $D $COMMON_FLAGS > "$L/video_${OUT%.mp4}.log" 2>&1
  echo "rc=$?  size $(ls -lh $V/$OUT 2>/dev/null | awk '{print $5}')"
  grep -aE "^ *Episode" "$L/video_${OUT%.mp4}.log" | tail -3
done
echo TRIPTYCH_DONE

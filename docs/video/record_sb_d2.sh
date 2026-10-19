#!/bin/bash
# arm_sb counterpart of model_arm_d_d2_robustness.mp4: same seed (1253), d=2, same
# envelope; only the model differs.
cd /home/kwadracik/Batchelor/sim2real-kinova-ppo-rl-in_robosuite
V=docs/video; L=logs/board_6dof/stdout
START_POSE="0.0,1.4692,-2.2410,3.1403,-2.0200,3.1416"
CK=logs/board_6dof/arm_sb/checkpoints/ppo_board_90123552_steps.zip
timeout 3600 conda run --no-capture-output -n robosuite \
  python training/watch_board.py --record "$V/model_arm_sb_d2_comparison.mp4" \
  --seeds 1253 --action-delay 2 --model $CK \
  --robot Kinova3_6DOF --horizon 2000 --kinova-limits --acc-limits --acc-mode scale \
  --estop-config training/configs/ppo_board_6dof_estop_hw.yaml \
  --spot-body --spot-front-cam --base-x-offset=-0.34 --init-qpos=$START_POSE \
  --fps 20 --width 1280 --height 720 --camera board_view \
  > "$L/video_model_arm_sb_d2.log" 2>&1
echo "rc=$?  size $(ls -lh $V/model_arm_sb_d2_comparison.mp4 2>/dev/null | awk '{print $5}')"
grep -aE "^ *Episode|Video saved" "$L/video_model_arm_sb_d2.log" | tail -3
echo SB_D2_DONE

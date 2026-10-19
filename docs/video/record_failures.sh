#!/bin/bash
# Re-record arm_d failures at d=0 (seeds 1271, 1275) after the OOM fix.
cd /home/kwadracik/Batchelor/sim2real-kinova-ppo-rl-in_robosuite
V=docs/video; L=logs/board_6dof/stdout
# No `ps | grep watch_board` wait loop: it matches the harness's own shell
# (2026-10-10). Memory is O(1) now anyway.
echo "[failures] RAM available: $(free -g | awk 'NR==2{print $7}') GB"
START_POSE="0.0,1.4692,-2.2410,3.1403,-2.0200,3.1416"
CK=logs/board_6dof/arm_d/checkpoints/ppo_board_98123552_steps.zip
timeout 3600 conda run --no-capture-output -n robosuite \
  python training/watch_board.py --record "$V/model_arm_d_failures.mp4" \
  --seeds 1271,1275 --model $CK \
  --robot Kinova3_6DOF --horizon 2000 --kinova-limits --acc-limits --acc-mode scale \
  --estop-config training/configs/ppo_board_6dof_estop_hw.yaml \
  --spot-body --spot-front-cam --base-x-offset=-0.34 --init-qpos=$START_POSE \
  --fps 20 --width 1280 --height 720 --camera board_view \
  > "$L/video_model_arm_d_failures.log" 2>&1
echo "rc=$?  size $(ls -lh $V/model_arm_d_failures.mp4 2>/dev/null | awk '{print $5}')"
echo "peak RSS during the run: see below"
grep -aE "^ *Episode|Video saved" "$L/video_model_arm_d_failures.log" | tail -3
echo FAILURES_DONE

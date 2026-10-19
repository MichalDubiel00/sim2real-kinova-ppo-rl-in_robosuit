#!/bin/bash
# Preemption-tolerant runner for one arm.
# Progress = the HIGHEST checkpoint in GCS. `--timesteps` is ADDITIONAL steps
# (reset_num_timesteps=False), so pass the remainder, not the target.
# Resuming is safe only because every curriculum in these configs is past its end;
# an active curriculum needs an explicit *_anneal_origin (docs 6.32).
set -uo pipefail
ARM="${ARM:?brak ARM}"              # k | s | sm | sb
BUCKET="${BUCKET:?brak BUCKET}"
TARGET="${TARGET:-90123552}"        # 86 123 552 + 4M
REPO="${REPO:-/opt/kinova/repo}"
# Start checkpoint from metadata (K/S/SM/SB from e6ab 86.12M, D from arm_sb 90.12M).
START_NAZWA="${START_NAZWA:-ppo_board_86123552_steps.zip}"
START="gs://$BUCKET/start/$START_NAZWA"
LOG="$REPO/logs/board_6dof/arm_$ARM"
CFG="$REPO/training/configs/arm_$ARM.yaml"
GS="gs://$BUCKET/arms/arm_$ARM"

# Shut the VM down after reaching the target, or Restart=always keeps it alive and
# billing (arm_s, 09.10). Only on success; failures are retried by systemd.
zakoncz_i_zgas() {
  echo "[$ARM] target reached - shutting the instance down"
  echo "${1:-$TARGET}" | gsutil cp - "$GS/KONIEC" 2>/dev/null
  shutdown -h +1 "ramie $ARM ukonczone" 2>/dev/null || poweroff 2>/dev/null
  exit 0
}

cd "$REPO"
mkdir -p "$LOG/checkpoints" "$REPO/logs/tensorboard" "$REPO/logs/stdout" /opt/kinova/start

# 1. Restore state from GCS (the disk may be new after preemption)
gsutil -q -m rsync -r "$GS/checkpoints" "$LOG/checkpoints" 2>/dev/null || true
if gsutil -q stat "$GS/KONIEC" 2>/dev/null; then
  echo "[$ARM] arm marked as FINISHED"
  shutdown -h +1 "ramie $ARM juz ukonczone" 2>/dev/null || poweroff 2>/dev/null
  exit 0
fi

# 2. Highest checkpoint = current step
LATEST=$(ls "$LOG/checkpoints"/ppo_board_*_steps.zip 2>/dev/null \
         | sed 's/.*ppo_board_\([0-9]*\)_steps.zip/\1 &/' | sort -n | tail -1)
if [ -n "$LATEST" ]; then
  CUR=${LATEST%% *}; RESUME=${LATEST#* }
  echo "[$ARM] resuming from checkpoint: step $CUR"
else
  CUR="${START_KROK:-86123552}"
  RESUME="/opt/kinova/start/$START_NAZWA"
  [ -f "$RESUME" ] || gsutil cp "$START" "$RESUME" || exit 1
  echo "[$ARM] start from $START_NAZWA: step $CUR"
fi

REMAIN=$(( TARGET - CUR ))
if [ "$REMAIN" -le 0 ]; then
  echo "[$ARM] target $TARGET reached (step $CUR)"
  zakoncz_i_zgas "$CUR"
fi
echo "[$ARM] $REMAIN steps left to $TARGET"

# 3. Background sync every 5 min + on exit (preemption gives ~30 s)
sync_all() {
  gsutil -q -m rsync -r "$LOG/checkpoints" "$GS/checkpoints" 2>/dev/null
  gsutil -q -m rsync -r "$REPO/logs/tensorboard" "$GS/tensorboard" 2>/dev/null
  gsutil -q cp "$REPO/logs/stdout/arm_$ARM.log" "$GS/stdout.log" 2>/dev/null
  [ -f "$LOG/eval/evaluations.npz" ] && gsutil -q cp "$LOG/eval/evaluations.npz" "$GS/evaluations.npz"
  return 0
}
( while true; do sleep 300; sync_all; done ) & SYNC_PID=$!
trap 'sync_all; kill $SYNC_PID 2>/dev/null' EXIT INT TERM

# 4. Training
python training/train_board.py --config "$CFG" --resume "$RESUME" --timesteps "$REMAIN" \
  >> "$REPO/logs/stdout/arm_$ARM.log" 2>&1
RC=$?
sync_all
echo "[$ARM] train_board exited with code $RC"
# Exit 0 = target reached. Anything else (preemption/SIGTERM): systemd restarts and
# the remainder is recomputed from the checkpoint.
if [ $RC -eq 0 ]; then zakoncz_i_zgas; fi
exit $RC

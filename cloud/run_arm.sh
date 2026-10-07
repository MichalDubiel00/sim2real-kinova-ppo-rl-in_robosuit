#!/bin/bash
# Odporny na preempcje runner jednego ramienia.
#
# ZASADA: jedynym zrodlem prawdy o postepie jest NAJWYZSZY checkpoint w GCS.
# Po ubiciu instancji skrypt wstaje, sciaga checkpointy, liczy ILE JESZCZE i wznawia.
# `--timesteps` w train_board to kroki DODATKOWE (SB3 reset_num_timesteps=False),
# wiec trzeba podac RESZTE, nie cel.
#
# ⚠ WZNAWIANIE JEST BEZPIECZNE TYLKO DLATEGO, ze wszystkie kurikula w tych configach
# sa PO swoim koncu (action_rate_anneal_origin 50223552 przy > 86M krokow => frac > 1,
# kara stoi na 0.3). Config z AKTYWNYM kurikulum wymaga jawnego `*_anneal_origin`,
# inaczej kazde wznowienie liczy postep od nowa — pulapka z docs 6.32.
set -uo pipefail
ARM="${ARM:?brak ARM}"              # k | s | sm | sb
BUCKET="${BUCKET:?brak BUCKET}"
TARGET="${TARGET:-90123552}"        # 86 123 552 + 4M
REPO="${REPO:-/opt/kinova/repo}"
START="gs://$BUCKET/start/ppo_board_86123552_steps.zip"
LOG="$REPO/logs/board_6dof/arm_$ARM"
CFG="$REPO/training/configs/arm_$ARM.yaml"
GS="gs://$BUCKET/arms/arm_$ARM"

cd "$REPO"
mkdir -p "$LOG/checkpoints" "$REPO/logs/tensorboard" "$REPO/logs/stdout" /opt/kinova/start

# 1. Odtworzenie stanu z GCS (po preempcji dysk moze byc nowy)
gsutil -q -m rsync -r "$GS/checkpoints" "$LOG/checkpoints" 2>/dev/null || true
if gsutil -q stat "$GS/KONIEC" 2>/dev/null; then
  echo "[$ARM] ramie oznaczone jako ZAKONCZONE — nic do roboty"; exit 0
fi

# 2. Najwyzszy checkpoint = aktualny krok
LATEST=$(ls "$LOG/checkpoints"/ppo_board_*_steps.zip 2>/dev/null \
         | sed 's/.*ppo_board_\([0-9]*\)_steps.zip/\1 &/' | sort -n | tail -1)
if [ -n "$LATEST" ]; then
  CUR=${LATEST%% *}; RESUME=${LATEST#* }
  echo "[$ARM] wznowienie z checkpointu: krok $CUR"
else
  CUR=86123552
  RESUME=/opt/kinova/start/ppo_board_86123552_steps.zip
  [ -f "$RESUME" ] || gsutil cp "$START" "$RESUME" || exit 1
  echo "[$ARM] start od e6ab: krok $CUR"
fi

REMAIN=$(( TARGET - CUR ))
if [ "$REMAIN" -le 0 ]; then
  echo "[$ARM] cel $TARGET osiagniety (krok $CUR)"
  echo "$CUR" | gsutil cp - "$GS/KONIEC"; exit 0
fi
echo "[$ARM] zostalo $REMAIN krokow do $TARGET"

# 3. Synchronizacja w tle co 5 min + przy wyjsciu (preempcja daje ok. 30 s)
sync_all() {
  gsutil -q -m rsync -r "$LOG/checkpoints" "$GS/checkpoints" 2>/dev/null
  gsutil -q -m rsync -r "$REPO/logs/tensorboard" "$GS/tensorboard" 2>/dev/null
  gsutil -q cp "$REPO/logs/stdout/arm_$ARM.log" "$GS/stdout.log" 2>/dev/null
  [ -f "$LOG/eval/evaluations.npz" ] && gsutil -q cp "$LOG/eval/evaluations.npz" "$GS/evaluations.npz"
  return 0
}
( while true; do sleep 300; sync_all; done ) & SYNC_PID=$!
trap 'sync_all; kill $SYNC_PID 2>/dev/null' EXIT INT TERM

# 4. Trening
python training/train_board.py --config "$CFG" --resume "$RESUME" --timesteps "$REMAIN" \
  >> "$REPO/logs/stdout/arm_$ARM.log" 2>&1
RC=$?
sync_all
echo "[$ARM] train_board zakonczony kodem $RC"
# Kod 0 = doszlo do celu. Inny kod (np. preempcja/SIGTERM) = systemd wznowi,
# a skrypt policzy reszte od nowa z checkpointu.
if [ $RC -eq 0 ]; then echo "$TARGET" | gsutil cp - "$GS/KONIEC"; fi
exit $RC

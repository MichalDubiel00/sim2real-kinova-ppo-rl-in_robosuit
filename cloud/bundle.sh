#!/bin/bash
# Pakowanie kodu i checkpointu startowego do GCS (uruchamiane LOKALNIE, raz).
# Repozytorium nie ma zdalnego gita, wiec VM nie moze go sklonowac — jedzie tarball.
set -euo pipefail
BUCKET="${1:?uzycie: cloud/bundle.sh NAZWA_BUCKETA}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="logs/board_6dof/spot_e6ab/checkpoints/ppo_board_86123552_steps.zip"
TMP="$(mktemp -d)"

cd "$REPO"
[ -f "$CKPT" ] || { echo "BRAK checkpointu startowego: $CKPT" >&2; exit 1; }

# Kod bez logow i bez .git (ok. 14 MB). my_assets JEST potrzebne — sa tam nasze XML-e.
tar czf "$TMP/kod.tar.gz" \
  --exclude='./logs' --exclude='./.git' --exclude='./docs/img' --exclude='./docs/video' \
  --exclude='__pycache__' --exclude='*.pyc' \
  envs training ros2_bridge my_assets docs/dokumentacja_projektu.md cloud

echo "rozmiar paczki: $(du -h "$TMP/kod.tar.gz" | cut -f1)"
gsutil cp "$TMP/kod.tar.gz" "gs://$BUCKET/kod.tar.gz"
gsutil cp "$CKPT" "gs://$BUCKET/start/ppo_board_86123552_steps.zip"
rm -rf "$TMP"
echo "GOTOWE. W gs://$BUCKET/ sa: kod.tar.gz, start/ppo_board_86123552_steps.zip"

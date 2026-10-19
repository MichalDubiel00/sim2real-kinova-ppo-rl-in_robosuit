#!/bin/bash
# Package code and the start checkpoint to GCS (run LOCALLY).
# Contents come from git (`git ls-files`), not a hand-written list: the list missed
# objects/ on 08.10. tar reads the working tree, so uncommitted fixes are included.
set -euo pipefail
BUCKET="${1:?uzycie: cloud/bundle.sh NAZWA_BUCKETA}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="logs/board_6dof/spot_e6ab/checkpoints/ppo_board_86123552_steps.zip"
TMP="$(mktemp -d)"

cd "$REPO"
[ -f "$CKPT" ] || { echo "MISSING start checkpoint: $CKPT" >&2; exit 1; }

# Skip docs/img (14 MB of PNG). --cached --others --exclude-standard: tracked AND new
# untracked files, honouring .gitignore (plain ls-files dropped training/configs/arm_d.yaml).
git ls-files -z --cached --others --exclude-standard | grep -zv '^docs/img/' > "$TMP/lista"
tar czf "$TMP/kod.tar.gz" --null -T "$TMP/lista"

# Check modules and assets training needs. List to a file, not `| grep -q`: pipefail +
# grep's early exit SIGPIPEs tar and reports a false miss.
tar tzf "$TMP/kod.tar.gz" > "$TMP/spis"
for m in envs/button_board.py envs/kinova3_6dof_robot.py \
         objects/button_object.py objects/custom_objects.py \
         my_assets/robots/kinova3_6dof/robot.xml my_assets/objects/xml/omniboard.xml \
         my_assets/objects/meshes/Omniboard.stl \
         training/train_board.py cloud/run_arm.sh; do
  grep -qx "$m" "$TMP/spis" || { echo "INCOMPLETE BUNDLE: missing $m" >&2; exit 1; }
done
# Every local arm config MUST be in the bundle, or systemd restarts the VM runner forever.
for cfg in training/configs/arm_*.yaml; do
  [ -e "$cfg" ] || continue
  grep -qx "$cfg" "$TMP/spis" || { echo "INCOMPLETE BUNDLE: missing $cfg" >&2; exit 1; }
done
# Symlinks with ABSOLUTE targets break on the VM (my_assets/objects/xml/meshes did).
while IFS= read -r l; do
  t="$(readlink "$l")" || continue
  case "$t" in /*) echo "ABSOLUTE SYMLINK: $l -> $t (will not survive on the VM)" >&2; exit 1;; esac
done < <(git ls-files -s | awk '$1=="120000"{ $1=$2=$3=""; sub(/^ +/,""); print }')
echo "bundle size: $(du -h "$TMP/kod.tar.gz" | cut -f1), files: $(tar tzf "$TMP/kod.tar.gz" | wc -l)"

gsutil cp "$TMP/kod.tar.gz" "gs://$BUCKET/kod.tar.gz"
gsutil cp "$CKPT" "gs://$BUCKET/start/ppo_board_86123552_steps.zip"
rm -rf "$TMP"
echo "DONE. gs://$BUCKET/ contains: kod.tar.gz, start/ppo_board_86123552_steps.zip"

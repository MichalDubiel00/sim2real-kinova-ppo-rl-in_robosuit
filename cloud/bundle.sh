#!/bin/bash
# Pakowanie kodu i checkpointu startowego do GCS (uruchamiane LOKALNIE).
#
# ZAWARTOSC PACZKI WYPROWADZONA Z GITA (`git ls-files`), nie z recznej listy katalogow.
# Reczna lista zawiodla 08.10: brakowalo `objects/` (importowane przez button_board.py),
# wiec cztery ramiona wstaly i padly na ModuleNotFoundError. Git wie, co jest kodem,
# a `logs/` i tak jest w .gitignore. Tar bierze tresc z DRZEWA ROBOCZEGO, wiec
# niezacommitowane poprawki tez jada.
set -euo pipefail
BUCKET="${1:?uzycie: cloud/bundle.sh NAZWA_BUCKETA}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="logs/board_6dof/spot_e6ab/checkpoints/ppo_board_86123552_steps.zip"
TMP="$(mktemp -d)"

cd "$REPO"
[ -f "$CKPT" ] || { echo "BRAK checkpointu startowego: $CKPT" >&2; exit 1; }

# docs/img pomijamy (14 MB PNG, na VM bezuzyteczne); reszta kodu jedzie w calosci.
git ls-files -z | grep -zv '^docs/img/' > "$TMP/lista"
tar czf "$TMP/kod.tar.gz" --null -T "$TMP/lista"

# Kontrola: moduly i zasoby, bez ktorych trening nie wstanie.
# Listing do pliku, NIE przez potok do `grep -q` — pipefail + wczesne wyjscie grepa
# daje tarowi SIGPIPE i kontrola zglasza falszywy brak (przejechalem sie na tym).
tar tzf "$TMP/kod.tar.gz" > "$TMP/spis"
for m in envs/button_board.py envs/kinova3_6dof_robot.py \
         objects/button_object.py objects/custom_objects.py \
         my_assets/robots/kinova3_6dof/robot.xml my_assets/objects/xml/omniboard.xml \
         my_assets/objects/meshes/Omniboard.stl \
         training/train_board.py cloud/run_arm.sh; do
  grep -qx "$m" "$TMP/spis" || { echo "PACZKA NIEPELNA: brak $m" >&2; exit 1; }
done
# Symlinki o ABSOLUTNYCH celach nie przeniosa sie na VM. `my_assets/objects/xml/meshes`
# wskazywal na /home/kwadracik/Batchelor/robosuite/... i dzialal lokalnie tylko przez
# drugi symlink katalogu; na czystej maszynie omniboard.xml nie znalazlby meshy.
while IFS= read -r l; do
  t="$(readlink "$l")" || continue
  case "$t" in /*) echo "SYMLINK ABSOLUTNY: $l -> $t (nie przeniesie sie na VM)" >&2; exit 1;; esac
done < <(git ls-files -s | awk '$1=="120000"{ $1=$2=$3=""; sub(/^ +/,""); print }')
echo "rozmiar paczki: $(du -h "$TMP/kod.tar.gz" | cut -f1), plikow: $(tar tzf "$TMP/kod.tar.gz" | wc -l)"

gsutil cp "$TMP/kod.tar.gz" "gs://$BUCKET/kod.tar.gz"
gsutil cp "$CKPT" "gs://$BUCKET/start/ppo_board_86123552_steps.zip"
rm -rf "$TMP"
echo "GOTOWE. W gs://$BUCKET/ sa: kod.tar.gz, start/ppo_board_86123552_steps.zip"

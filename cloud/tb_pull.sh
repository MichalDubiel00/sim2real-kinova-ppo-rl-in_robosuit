#!/bin/bash
# Sciagniecie krzywych TensorBoard z czterech ramion i uruchomienie podgladu lokalnie.
#
# DLACZEGO LOKALNIE, a nie `tensorboard --logdir gs://...`: obsluga gs:// w TB idzie
# przez tensorflow, a my TF swiadomie blokujemy (`sys.modules["tensorflow"] = None`,
# inaczej segfault w PPO.load). Wiec: rsync do katalogu lokalnego i podglad z dysku.
set -uo pipefail
BUCKET="${1:-sim2real_robosuite}"
DST="${2:-logs/tensorboard_chmura}"
mkdir -p "$DST"
for ARM in k s sm sb; do
  gsutil -q -m rsync -r "gs://$BUCKET/arms/arm_$ARM/tensorboard" "$DST/arm_$ARM" 2>/dev/null \
    && echo "arm_$ARM: zsynchronizowane" || echo "arm_$ARM: jeszcze nic nie ma"
done
echo
echo "Biegi w pobranych danych (po preempcji kazde wznowienie tworzy NOWY bieg _1, _2, ...):"
find "$DST" -name "events*" | sed "s#$DST/##; s#/events.*##" | sort | uniq -c
echo
echo "Podglad:  tensorboard --logdir $DST --port 6006"

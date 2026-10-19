#!/bin/bash
# Pull TensorBoard curves from the four arms and view them locally.
# Not `--logdir gs://`: gs:// support goes through TensorFlow, which we block.
set -uo pipefail
BUCKET="${1:-sim2real_robosuite}"
DST="${2:-logs/tensorboard_chmura}"
mkdir -p "$DST"
for ARM in k s sm sb d; do
  mkdir -p "$DST/arm_$ARM"
  gsutil -q -m rsync -r "gs://$BUCKET/arms/arm_$ARM/tensorboard" "$DST/arm_$ARM" 2>/dev/null \
    && echo "arm_$ARM: synced" || echo "arm_$ARM: nothing yet"
done
echo
echo "Runs in the downloaded data (after preemption every resume creates a NEW run _1, _2, ...):"
find "$DST" -name "events*" | sed "s#$DST/##; s#/events.*##" | sort | uniq -c
echo
echo "View:  tensorboard --logdir $DST --port 6006"

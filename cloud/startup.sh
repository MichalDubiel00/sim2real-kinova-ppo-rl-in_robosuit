#!/bin/bash
# GCP startup script: runs as root on EVERY boot, including after preemption, so it must
# be idempotent.
#
# Required instance metadata: arm=k|s|sm|sb, bucket=NAME, target-steps=90123552
set -uo pipefail
exec > >(tee -a /var/log/kinova-startup.log) 2>&1
echo "=== startup $(date -Is) ==="

md() { curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
ARM="$(md arm)"; BUCKET="$(md bucket)"; TARGET="$(md target-steps || echo 90123552)"
START_NAZWA="$(md start-ckpt || echo ppo_board_86123552_steps.zip)"
START_KROK="$(md start-step || echo 86123552)"
[ -n "$ARM" ] && [ -n "$BUCKET" ] || { echo "MISSING arm/bucket metadata"; exit 1; }
echo "arm=$ARM bucket=$BUCKET target=$TARGET"

CONDA=/opt/miniconda
ENVP=$CONDA/envs/robosuite
REPO=/opt/kinova/repo

# 1. System packages. No rendering, but libosmesa6 is cheap insurance.
BRAKUJE=""
# libgl1, libglib2.0-0: needed by opencv-python (`import robosuite` fails without
# libGL.so.1). build-essential: evdev (via pynput) builds from sdist.
for pkt in curl tar libosmesa6 build-essential libgl1 libglib2.0-0; do
  dpkg -s "$pkt" >/dev/null 2>&1 || BRAKUJE="$BRAKUJE $pkt"
done
if [ -n "$BRAKUJE" ]; then
  echo "installing packages:$BRAKUJE"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $BRAKUJE >/dev/null
fi

# 1b. gsutil is not guaranteed on GCE images; without it no checkpoint reaches GCS.
if ! command -v gsutil >/dev/null 2>&1; then
  echo "gsutil missing - installing google-cloud-cli"
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq apt-transport-https ca-certificates gnupg >/dev/null
  curl -sS https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq google-cloud-cli >/dev/null
fi
command -v gsutil >/dev/null 2>&1 || { echo "NO gsutil after installation - aborting"; exit 1; }
# Test WRITE access before training (usual cause: missing --scopes=devstorage.read_write).
echo "test-$(date -Is)" | gsutil cp - "gs://$BUCKET/arms/.zapis-ok-$ARM" \
  || { echo "NO WRITE ACCESS to gs://\$BUCKET. Two independent causes, both required:"
       echo "  1) instance scope: --scopes=https://www.googleapis.com/auth/devstorage.read_write"
       echo "  2) service account IAM (our mistake on the first start, 08.10):"
       echo "     gcloud storage buckets add-iam-policy-binding gs://\$BUCKET \\"
       echo "       --member=serviceAccount:<NUMBER>-compute@developer.gserviceaccount.com \\"
       echo "       --role=roles/storage.objectAdmin"
       exit 1; }
echo "bucket write OK"

# 2. Miniconda + PINNED env. numpy < 2 (robosuite 1.5.1); CPU torch (MLP 256x256,
#    MuJoCo stepping is the bottleneck).
if [ ! -x "$CONDA/bin/conda" ]; then
  curl -sSL -o /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/mc.sh -b -p "$CONDA" >/dev/null
fi
if [ ! -x "$ENVP/bin/python" ]; then
  # conda-forge with --override-channels: the default channels require ToS acceptance
  # and fail non-interactively (CondaToSNonInteractiveError). Output not silenced.
  "$CONDA/bin/conda" create -y -n robosuite python=3.10 -c conda-forge --override-channels
  [ -x "$ENVP/bin/python" ] || { echo "FAILED to create the conda env - aborting"; exit 1; }
fi

# Condition is a WORKING import, not a python binary (pip once failed halfway).
if ! "$ENVP/bin/python" -c "import robosuite, mujoco, stable_baselines3, torch" >/dev/null 2>&1; then
  echo "installing Python packages"
  "$ENVP/bin/pip" install -q --upgrade pip
  # CPU torch: GPU gives nothing here.
  "$ENVP/bin/pip" install -q torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu \
    || "$ENVP/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu \
    || { echo "torch installation failed - aborting"; exit 1; }
  # Versions copied from the working local env (pip freeze); numpy < 2.
  "$ENVP/bin/pip" install -q \
     numpy==1.26.4 scipy==1.15.3 mujoco==3.3.7 robosuite==1.5.1 \
     stable-baselines3==2.3.2 gymnasium==0.29.1 tensorboard==2.20.0 \
     pynput==1.8.1 evdev==1.6.1 pyyaml \
    || { echo "pip install failed - aborting"; exit 1; }
fi
"$ENVP/bin/python" -c "import robosuite,mujoco,stable_baselines3,numpy;print('env OK',
 robosuite.__version__, mujoco.__version__, stable_baselines3.__version__, numpy.__version__)" \
  || { echo "env incomplete after installation - aborting"; exit 1; }

# 3. Code from GCS (source of truth); logs live outside the tarball.
mkdir -p "$REPO" /opt/kinova/start
gsutil -q cp "gs://$BUCKET/kod.tar.gz" /tmp/kod.tar.gz || exit 1
tar xzf /tmp/kod.tar.gz -C "$REPO"
chmod +x "$REPO"/cloud/*.sh

# 4. Service. Restart=always: after preemption run_arm.sh resumes from the highest
#    checkpoint. Short TimeoutStopSec: preemption gives ~30 s to sync.
cat > /etc/systemd/system/kinova-arm.service <<UNIT
[Unit]
Description=Trening ramienia $ARM (ButtonBoard 6DOF)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=ARM=$ARM
Environment=BUCKET=$BUCKET
Environment=TARGET=$TARGET
Environment=START_NAZWA=$START_NAZWA
Environment=START_KROK=$START_KROK
Environment=REPO=$REPO
Environment=MUJOCO_GL=osmesa
Environment=OMP_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
Environment=PATH=$ENVP/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
WorkingDirectory=$REPO
ExecStart=$REPO/cloud/run_arm.sh
Restart=always
RestartSec=60
TimeoutStopSec=25
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now kinova-arm.service
echo "=== startup done, service started ==="

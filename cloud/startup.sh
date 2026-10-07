#!/bin/bash
# GCP startup-script — wykonywany jako root PRZY KAZDYM BOOCIE, takze po preempcji.
# Musi byc idempotentny: instaluje tylko to, czego brakuje, i wstaje w stan "trenuj dalej".
#
# Metadane instancji (wymagane): arm=k|s|sm|sb, bucket=NAZWA, target-steps=90123552
set -uo pipefail
exec > >(tee -a /var/log/kinova-startup.log) 2>&1
echo "=== startup $(date -Is) ==="

md() { curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
ARM="$(md arm)"; BUCKET="$(md bucket)"; TARGET="$(md target-steps || echo 90123552)"
[ -n "$ARM" ] && [ -n "$BUCKET" ] || { echo "BRAK metadanych arm/bucket"; exit 1; }
echo "ramie=$ARM bucket=$BUCKET cel=$TARGET"

CONDA=/opt/miniconda
ENVP=$CONDA/envs/robosuite
REPO=/opt/kinova/repo

# 1. Zaleznosci systemowe. GL NIE jest potrzebny (use_camera_obs=False,
#    has_offscreen_renderer=False), ale libosmesa6 jest tani i zdejmuje ryzyko,
#    gdyby cokolwiek chcialo zainicjowac kontekst.
if ! dpkg -s libosmesa6 >/dev/null 2>&1; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl tar libosmesa6 >/dev/null
fi

# 2. Miniconda + srodowisko z PRZYPIETYMI wersjami.
#    numpy MUSI byc < 2 (robosuite 1.5.1), torch w wersji CPU — siec to MLP 256x256,
#    GPU nic nie daje, a waskim gardlem jest stepowanie MuJoCo na CPU.
if [ ! -x "$CONDA/bin/conda" ]; then
  curl -sSL -o /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/mc.sh -b -p "$CONDA" >/dev/null
fi
if [ ! -x "$ENVP/bin/python" ]; then
  "$CONDA/bin/conda" create -y -q -n robosuite python=3.10 >/dev/null
  "$ENVP/bin/pip" install -q --upgrade pip
  "$ENVP/bin/pip" install -q torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu \
    || "$ENVP/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
  "$ENVP/bin/pip" install -q \
     numpy==1.26.4 scipy==1.15.3 mujoco==3.3.7 robosuite==1.5.1 \
     stable-baselines3==2.3.2 tensorboard pyyaml
fi
"$ENVP/bin/python" -c "import robosuite,mujoco,stable_baselines3,numpy;print('env OK',
 robosuite.__version__, mujoco.__version__, stable_baselines3.__version__, numpy.__version__)" || exit 1

# 3. Kod z GCS. Zrodlem prawdy dla KODU jest bucket; logi zyja poza tarballem,
#    wiec ponowne rozpakowanie przy kazdym boocie nic nie niszczy.
mkdir -p "$REPO" /opt/kinova/start
gsutil -q cp "gs://$BUCKET/kod.tar.gz" /tmp/kod.tar.gz || exit 1
tar xzf /tmp/kod.tar.gz -C "$REPO"
chmod +x "$REPO"/cloud/*.sh

# 4. Usluga. Restart=always => po preempcji i ponownym starcie instancji
#    run_arm.sh policzy reszte krokow z najwyzszego checkpointu i wznowi.
#    TimeoutStopSec krotki, bo preempcja daje ok. 30 s na zrzut do GCS.
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
echo "=== startup gotowy, usluga wystartowana ==="

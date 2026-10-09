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
BRAKUJE=""
# libgl1 i libglib2.0-0: ciagnie je opencv-python (zaleznosc robosuite) i bez nich
# `import robosuite` wywala ImportError: libGL.so.1 — tak padl czwarty start 08.10,
# mimo ze nic nie renderujemy. build-essential: evdev (przez pynput) jest tylko sdist
# i kompiluje sie u klienta, wymaga gcc i naglowka linux/input.h.
for pkt in curl tar libosmesa6 build-essential libgl1 libglib2.0-0; do
  dpkg -s "$pkt" >/dev/null 2>&1 || BRAKUJE="$BRAKUJE $pkt"
done
if [ -n "$BRAKUJE" ]; then
  echo "instaluje pakiety:$BRAKUJE"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $BRAKUJE >/dev/null
fi

# 1b. gsutil. Obrazy GCE zwykle maja google-cloud-cli, ale NIE jest to gwarantowane,
#     a bez gsutil kazde ramie wstanie i nie zapisze ANI JEDNEGO checkpointu do GCS —
#     awaria cicha, kosztujaca caly bieg. Dlatego sprawdzamy wprost.
if ! command -v gsutil >/dev/null 2>&1; then
  echo "gsutil nieobecny — instaluje google-cloud-cli"
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq apt-transport-https ca-certificates gnupg >/dev/null
  curl -sS https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq google-cloud-cli >/dev/null
fi
command -v gsutil >/dev/null 2>&1 || { echo "BRAK gsutil po instalacji — przerywam"; exit 1; }
# Test ZAPISU do bucketa, zanim zaczniemy liczyc. Najczestszy blad to brak
# --scopes=devstorage.read_write przy tworzeniu instancji: runner wstaje i milczy.
echo "test-$(date -Is)" | gsutil cp - "gs://$BUCKET/arms/.zapis-ok-$ARM" \
  || { echo "BRAK PRAWA ZAPISU do gs://\$BUCKET. Dwie niezalezne przyczyny, obie konieczne:"
       echo "  1) zakres instancji: --scopes=https://www.googleapis.com/auth/devstorage.read_write"
       echo "  2) IAM konta uslugi (to byl nasz blad przy pierwszym starcie 08.10):"
       echo "     gcloud storage buckets add-iam-policy-binding gs://\$BUCKET \\"
       echo "       --member=serviceAccount:<NUMER>-compute@developer.gserviceaccount.com \\"
       echo "       --role=roles/storage.objectAdmin"
       exit 1; }
echo "zapis do bucketa OK"

# 2. Miniconda + srodowisko z PRZYPIETYMI wersjami.
#    numpy MUSI byc < 2 (robosuite 1.5.1), torch w wersji CPU — siec to MLP 256x256,
#    GPU nic nie daje, a waskim gardlem jest stepowanie MuJoCo na CPU.
if [ ! -x "$CONDA/bin/conda" ]; then
  curl -sSL -o /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/mc.sh -b -p "$CONDA" >/dev/null
fi
if [ ! -x "$ENVP/bin/python" ]; then
  # KANAL conda-forge Z --override-channels, NIE domyslny. Conda wymaga dzis jawnej
  # akceptacji Terms of Service dla repo.anaconda.com/pkgs/{main,r} i w trybie
  # nieinteraktywnym KONCZY SIE BLEDEM (CondaToSNonInteractiveError) — tak padl
  # pierwszy start 08.10. conda-forge nie ma tej bramki ani watpliwosci licencyjnych.
  # Wyjscia NIE wyciszamy: to krok, ktory najlatwiej pada, a /dev/null ukryl blad.
  "$CONDA/bin/conda" create -y -n robosuite python=3.10 -c conda-forge --override-channels
  [ -x "$ENVP/bin/python" ] || { echo "NIE UDALO SIE utworzyc srodowiska conda — przerywam"; exit 1; }
fi

# Warunkiem jest DZIALAJACY import, nie istnienie pliku python. Przy trzecim starcie
# 08.10 srodowisko istnialo, ale pip padl w polowie (evdev) — warunek "czy jest python"
# pominalby instalacje i bieg wstalby bez robosuite.
if ! "$ENVP/bin/python" -c "import robosuite, mujoco, stable_baselines3, torch" >/dev/null 2>&1; then
  echo "doinstalowuje pakiety Pythona"
  "$ENVP/bin/pip" install -q --upgrade pip
  # torch w wersji CPU: siec to MLP 256x256, GPU nic nie daje, a waskim gardlem
  # jest stepowanie MuJoCo na CPU.
  "$ENVP/bin/pip" install -q torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu \
    || "$ENVP/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu \
    || { echo "instalacja torch nie przeszla — przerywam"; exit 1; }
  # Wersje przepisane z DZIALAJACEGO srodowiska lokalnego (pip list --format=freeze),
  # nie z pamieci. numpy MUSI byc < 2 (robosuite 1.5.1).
  "$ENVP/bin/pip" install -q \
     numpy==1.26.4 scipy==1.15.3 mujoco==3.3.7 robosuite==1.5.1 \
     stable-baselines3==2.3.2 gymnasium==0.29.1 tensorboard==2.20.0 \
     pynput==1.8.1 evdev==1.6.1 pyyaml \
    || { echo "pip install nie przeszedl — przerywam"; exit 1; }
fi
"$ENVP/bin/python" -c "import robosuite,mujoco,stable_baselines3,numpy;print('env OK',
 robosuite.__version__, mujoco.__version__, stable_baselines3.__version__, numpy.__version__)" \
  || { echo "srodowisko niekompletne po instalacji — przerywam"; exit 1; }

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

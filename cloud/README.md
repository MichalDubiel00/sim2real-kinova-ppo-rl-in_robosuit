# Kampania chmurowa: cztery ramiona kontrolne

Po co chmura: **nie po szybkość, po biegi kontrolne.** Największa dziura metodologiczna
kampanii 6.43–6.47 to brak ramion odniesienia — każde „ulepszenie" było jednocześnie
kolejnymi milionami kroków, więc poprawy nie dawało się przypisać (docs 6.46: kolizje
20.9 → 8.6 przy p = 0.047, ale biegu „te same 4M bez kary" nigdy nie zrobiono).
Cztery ramiona puszczone **równolegle** z tego samego checkpointu to usuwają.

## Ramiona

Wszystkie: **+4M kroków** z `spot_e6ab/checkpoints/ppo_board_86123552_steps.zip`
(model na sprzęt, n=100: 100/100, 15.2 s, kolizje 7.0, misuse 0.0).
Cel absolutny: **90 123 552** kroków.

| ramię | config | zmiana wobec e6ab | po co |
|---|---|---|---|
| **K** | `arm_k.yaml` | **żadna** | kontrola: ile daje samo 4M kroków |
| **S** | `arm_s.yaml` | `joint_acc_limit_mode: scale` | poprawka wierności — firmware skaluje proporcjonalnie (docs 6.50) |
| **SM** | `arm_sm.yaml` | `scale` + limity **4.0/8.0** | zgodność przez konstrukcję: trening ciaśniej niż sprzęt, żeby firmware nie przycinał wcale |
| **SB** | `arm_sb.yaml` | `scale` + kary `acc_*` = 0 | czy kary liczące tętnienie (docs 6.48) nie szkodzą |

**SM ocenia się przy 5.2/10.0**, nie przy 4.0/8.0 — inaczej mierzy się inną scenę
(ta sama pułapka co `base_x_offset` w 6.42).

## Uruchomienie

```bash
# 0. raz: bucket
gsutil mb -l europe-west3 gs://NAZWA-BUCKETA

# 1. raz: kod (~14 MB) + checkpoint startowy (2.3 MB) do GCS
cloud/bundle.sh NAZWA-BUCKETA

# 2. cztery instancje spot, po jednej na ramię
for ARM in k s sm sb; do
  gcloud compute instances create kinova-arm-$ARM \
    --zone=europe-west3-a \
    --machine-type=e2-standard-8 \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --boot-disk-size=40GB \
    --image-family=debian-12 --image-project=debian-cloud \
    --scopes=https://www.googleapis.com/auth/devstorage.read_write \
    --metadata=arm=$ARM,bucket=NAZWA-BUCKETA,target-steps=90123552 \
    --metadata-from-file=startup-script=cloud/startup.sh
done
```

`--scopes=devstorage.read_write` jest **konieczny** — bez niego `gsutil` na VM nie ma prawa
do bucketa i runner wstanie, ale nic nie zapisze.

## Preempcja

Instancja spot może zostać ubita w każdej chwili. Zabezpieczenie jest trzystopniowe:

1. checkpoint co **100 000 kroków** (`checkpoint_freq` w configu) — maksymalna strata,
2. synchronizacja do GCS co **5 min** plus zrzut przy SIGTERM (`trap` w `run_arm.sh`;
   GCE daje ok. 30 s, `TimeoutStopSec=25`),
3. `--instance-termination-action=STOP` + `Restart=always` — po ponownym starcie
   instancji `run_arm.sh` czyta **najwyższy checkpoint z GCS**, liczy ile zostało
   i wznawia. Nie trzeba niczego pamiętać między bootami.

Ubitą instancję trzeba wystartować ręcznie (`gcloud compute instances start kinova-arm-s
--zone=...`) albo z MIG-a z autohealingiem. Sprawdzanie: `gcloud compute instances list`.

> ⚠ **Wznawianie jest bezpieczne tylko dlatego, że wszystkie kurikula w tych configach są
> po swoim końcu** (`action_rate_anneal_origin: 50223552` przy > 86M kroków ⇒ `frac > 1`,
> kara stoi na docelowym 0.3). Config z **aktywnym** kurikulum wymaga jawnego
> `*_anneal_origin`, inaczej każde wznowienie liczy postęp od nowa — pułapka z docs 6.32,
> która raz już przejechała bieg `hwlim_curr`.

## Czego NIE zmieniać

- **`n_envs: 8`.** Przy 32 rdzeniach kusi 32 środowiska, ale to zmienia efektywny batch
  i dynamikę uczenia, a **cała nasza historia — 86M kroków i wszystkie testy na parach —
  jest przy 8**. Cztery ramiona po 8 env są warte więcej niż jedno po 32, i dokładnie po to
  bierzemy chmurę.
- `n_steps`, `batch_size`, `learning_rate`, `log_std_max`, `target_kl` — jak wyżej.
- Nagrania (`watch_board.py`) zostają lokalnie: wymagają renderowania, a ewaluacje nie
  (`use_camera_obs=False`).

## Postęp i odbiór wyników

```bash
# postęp (najwyższy checkpoint każdego ramienia)
for ARM in k s sm sb; do
  echo -n "arm_$ARM: "
  gsutil ls gs://NAZWA-BUCKETA/arms/arm_$ARM/checkpoints/ 2>/dev/null \
    | sed 's/.*ppo_board_\([0-9]*\)_steps.zip/\1/' | sort -n | tail -1
done

# ściągnięcie checkpointów do oceny lokalnej
for ARM in k s sm sb; do
  mkdir -p logs/board_6dof/arm_$ARM/checkpoints
  gsutil -m rsync -r gs://NAZWA-BUCKETA/arms/arm_$ARM/checkpoints \
    logs/board_6dof/arm_$ARM/checkpoints
done
```

Ocena **lokalnie**, tym samym `compare_models.py`, na tych samych ziarnach co e6ab,
w trybie docelowym `--acc-mode scale`:

```bash
POZA="0.0,1.4692,-2.2410,3.1403,-2.0200,3.1416"
python training/compare_models.py "arm_s=logs/.../arm_s/checkpoints/ppo_board_90123552_steps.zip" \
  --robot Kinova3_6DOF --episodes 100 --seed 1234 --horizon 2000 \
  --kinova-limits --acc-limits --acc-mode scale \
  --estop-config training/configs/ppo_board_6dof_estop_hw.yaml \
  --spot-body --spot-front-cam --base-x-offset=-0.34 --init-qpos="$POZA"
```

Odniesienie (e6ab, `--acc-mode scale`, n=40): 40/40, 14.6 s, kolizje **11.9** (mediana 6.5).
Przy `clip` było 5.2 — **nie mieszać tych dwóch liczb.**

## Koszt

Rachunek jest w godzinach-ramienia: 4 ramiona × 4M kroków. **Zmierzone 08.10 na
`e2-standard-8`: 164–180 kroków/s**, czyli ok. **6,5 h na ramię** i **~26 godzin-ramienia**
na czwórkę. Laptop robi ~250 kroków/s, więc pierwotny szacunek 18 h był o 45 % za niski —
vCPU w chmurze są słabsze od rdzeni laptopa. Ramiona liczą równolegle, więc zegarowo to ~6,5 h. Aktualnych cen `e2-standard-8` spot nie podaję z pamięci: trzeba je odczytać
w konsoli dla wybranego regionu. Instancje nie mają GPU świadomie — przy MLP 256×256
akcelerator nic nie wnosi, a wąskim gardłem jest stepowanie MuJoCo.

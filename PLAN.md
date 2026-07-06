# Plan implementacji: RL dla Kinova Gen 3 na Spocie — Robosuite

## Kontekst

Kinova Gen 3 zamontowana na Boston Dynamics Spot. Użytkownik podjeżdża Spotem do boardu z cylindrami i guzikami. Kinova autonomicznie (RL):
1. Zagląda do każdego cylindra wrist kamerą i robi zdjęcie środka
2. Naciska guziki Emergency Stop z odpowiednią siłą

Board nie jest zawsze w tej samej pozycji względem bazy Kinovy → potrzebna percepcja do lokalizacji.

---

## Zależności do zainstalowania (conda env: `robosuite`)

### Już zainstalowane
- `robosuite==1.5.1`
- `mujoco==3.3.7`
- `numpy==1.26.4`
- `scipy==1.15.3`
- `opencv-python==4.10.0.84`

### Do zainstalowania

```bash
conda activate robosuite

# RL
pip install stable-baselines3[extra]==2.3.2
pip install gymnasium==0.29.1

# Tensorboard (wykresy do pracy)
pip install tensorboard

# Imitation learning (opcjonalnie, BC jako warmstart)
pip install imitation

# Narzędzia
pip install imageio imageio-ffmpeg  # nagrywanie rolloutów
pip install transforms3d            # konwersje rotacji
```

---

## Struktura projektu

```
robosuite/
├── PLAN.md                        ← ten plik
├── envs/
│   ├── __init__.py
│   ├── cylinder_inspection.py     ← środowisko RL #1
│   └── button_press.py            ← środowisko RL #2
├── objects/
│   ├── custom_objects.py          ← już istnieje (OmniBoardObject)
│   ├── cylinder_object.py         ← CylinderObject do inspekcji
│   └── button_object.py           ← ButtonObject z symulacją sprężyny
├── my_assets/
│   ├── objects/
│   │   ├── xml/
│   │   │   ├── cylinder.xml       ← już istnieje
│   │   │   ├── omniboard.xml      ← cały board (5 cylindrów lub 5 guzików)
│   │   │   └── button.xml         ← guzik Emergency Stop ze sprężyną
│   │   └── meshes/
│   │       ├── 50mm_pipe v1.stl   ← już istnieje
│   │       ├── Emergency stop.stl ← już istnieje
│   │       └── Omniboard.stl      ← już istnieje
├── training/
│   ├── train_cylinder.py          ← skrypt treningowy PPO #1
│   ├── train_button.py            ← skrypt treningowy PPO #2
│   └── configs/
│       ├── ppo_cylinder.yaml      ← hiperparametry PPO
│       └── ppo_button.yaml
├── eval/
│   ├── evaluate.py                ← ewaluacja success rate
│   └── record_rollout.py          ← nagrywanie wideo dla pracy
└── perception/
    └── board_localizer.py         ← moduł percepcji (ArUco / depth)
```

---

## Faza 1: Środowiska MuJoCo / robosuite

### 1.1 `button.xml` — symulacja guzika Emergency Stop

Emergency Stop wymaga modelowania:
- **Sprężyna**: joint pryzmatyczny z `stiffness` i `damping`
- **Zakres ruchu**: ~15mm wciśnięcia
- **Siła aktywacji**: ~10–20N (do kalibracji z realnym guzikiem)
- **Sensor siły**: `<sensor type="force">` na geom kontaktu
- **Site docelowy**: miejsce gdzie EEF ma trafić

```xml
<!-- Kluczowe elementy button.xml -->
<joint name="button_slide" type="slide" axis="0 0 -1"
       range="0 0.015" stiffness="500" damping="20"/>
<sensor name="button_force" type="force" site="button_contact_site"/>
```

### 1.2 `CylinderInspection-v0`

**Klasa**: dziedziczy po `robosuite.environments.manipulation.ManipulationEnv`
**Robot**: `Kinova3` z `Robotiq85Gripper`
**Controller**: `OSC_POSE` (Operational Space Control — sterowanie pozą EEF)

**Przestrzeń akcji** (6D delta OSC):
```
[dx, dy, dz, droll, dpitch, dyaw]  ∈ [-1, 1]^6
```

**Przestrzeń obserwacji**:
```python
obs = {
    "robot0_eef_pos":           # [3] pozycja EEF w układzie bazy
    "robot0_eef_quat":          # [4] orientacja EEF (quaternion)
    "robot0_joint_pos":         # [7] kąty jointów
    "robot0_joint_vel":         # [7] prędkości jointów
    "target_pos_relative":      # [3] pozycja celu względem EEF
    "target_axis_relative":     # [3] oś cylindra w układzie EEF
    "camera_axis_error":        # [1] kąt między osią kamery a osią cylindra [rad]
    "distance_to_opening":      # [1] odległość EEF od otworu cylindra [m]
}
# Łącznie: ~26 wartości
```

**Funkcja nagrody**:
```python
r_alignment  = exp(-5.0 * camera_axis_angle_error)   # główna: kąt kamery vs oś cylindra
r_distance   = exp(-10.0 * distance_to_opening)       # podejście blisko otworu
r_stable     = 1.0 if aligned_for_N_steps else 0.0   # utrzymanie przez N=10 kroków
r_collision  = -1.0 if collision else 0.0             # kara za kolizję

reward = 2.0*r_alignment + 1.0*r_distance + 5.0*r_stable + r_collision
```

**Definicja sukcesu**:
- Kąt kamery vs oś cylindra < 5°
- Odległość EEF od otworu < 5cm
- Utrzymane przez 10 kolejnych kroków

**Randomizacja przy reset** (kluczowe dla sim2real):
- Pozycja cylindra: `±15cm` w XY, `±10°` w yaw
- Wysokość: `±5cm`
- Rozmiar cylindra: `±20%` (opcjonalnie)

### 1.3 `ButtonPress-v0`

**Przestrzeń obserwacji**: analogiczna do cylindra + `contact_force` z sensora

**Funkcja nagrody**:
```python
r_approach   = exp(-10.0 * distance_to_button)
r_alignment  = exp(-5.0 * eef_axis_vs_button_axis_error)
r_force      = 1.0 if F_min < contact_force < F_max else 0.0  # okno siły [10N, 25N]
r_activation = 5.0 if button_pressed else 0.0                 # guzik aktywowany
r_collision  = -1.0 if wrong_collision else 0.0

reward = 1.0*r_approach + 1.0*r_alignment + 2.0*r_force + 5.0*r_activation + r_collision
```

**Symulacja sprężyny guzika**:
- Joint pryzmatyczny z `stiffness=500 N/m` (do dostrojenia)
- Aktywacja gdy `joint_pos > 0.012m` (12mm wciśnięcia)

---

## Faza 2: Trening RL

### Algorytm: PPO (Proximal Policy Optimization)

Uzasadnienie wyboru:
- On-policy → prostsze debugowanie i stabilność
- Dobra zbieżność dla zadań manipulacyjnych z gęstą nagrodą
- Clip objective → bezpieczne kroki aktualizacji
- Łatwa paralelizacja (VecEnv) → szybszy trening

### Hiperparametry startowe

```yaml
# ppo_cylinder.yaml
algorithm: PPO
policy: MlpPolicy
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5
policy_kwargs:
  net_arch: [256, 256]
total_timesteps: 2_000_000
```

### Checkpointy i metryki (do pracy inżynierskiej)

- Zapis modelu co 100k kroków
- Tensorboard: `reward`, `success_rate`, `episode_length`, `entropy`
- Krzywe uczenia jako wykresy w pracy

---

## Faza 3: Ewaluacja w symulacji

### Testy do opisania w pracy

1. **Success rate vs liczba kroków treningowych** — wykres konwergencji
2. **Success rate vs kąt podejścia** — czy policy działa dla wszystkich 5 pozycji boardu
3. **Robustness test** — większa randomizacja niż podczas treningu (test generalizacji)
4. **Ablation study** — bez nagrody `r_alignment` vs z nią (uzasadnienie reward shapingu)

---

## Faza 4: Percepcja (lokalizacja boardu)

### Metoda: ArUco marker na boardzie

```
Board → marker ArUco (ID=0, rozmiar 10cm)
Wrist camera → cv2.aruco.detectMarkers()
Wynik: T_marker_in_camera → T_board_in_kinova_base
Pozycje cylindrów/guzików: znane geometrycznie względem markera
```

**Implementacja** (`perception/board_localizer.py`):
```python
def localize_board(image, camera_intrinsics, marker_size=0.1):
    """Zwraca pozycje wszystkich 5 cylindrów/guzików w układzie kamery."""
    ...
```

### Alternatywa markerless (jeśli wymagane)

- Intel RealSense D435 depth
- Hough Circle Transform na obrazie depth → detekcja otworów cylindrów
- Trudniejsze, ale ładniej wygląda w pracy jako wkład własny

---

## Faza 5: Sim2Real transfer

### Kluczowe kroki

1. **Kalibracja**:
   - Intrinsics wrist kamery (checkerboard)
   - Extrinsics: transformacja kamera → EEF Kinovy

2. **Policy deployment**:
   - Eksport policy: `model.policy.save("policy.zip")` (SB3 format)
   - Wrapper: `RealKinovaEnv` — taki sam interfejs jak sim, ale komendy idą przez Kortex API

3. **Kortex API (Python)**:
   ```python
   # Kinova Gen3 SDK
   from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
   # Wysyłanie komend prędkości / pozycji w Cartesian space
   ```

4. **Pętla wykonania**:
   ```
   while cylinders_remaining:
       obs = get_obs_from_real_robot()  # Kortex API + ArUco
       action = policy.predict(obs)
       send_action_to_kinova(action)    # Kortex API
       check_success()                  # kąt kamery OK? → foto
   ```

5. **Domain randomization** (do retreningu jeśli gap za duży):
   - Losowanie friction, mass cylindra
   - Losowanie lighting w symulacji (jeśli używamy pikseli)

### Spodziewane problemy sim2real

| Problem | Rozwiązanie |
|---------|-------------|
| Inertia/friction Kinovy różni się od MuJoCo | Dostroić model dynamiki lub użyć System ID |
| Opóźnienie w Kortex API ~10ms | Uwzględnić w symulacji (action delay wrapper) |
| Wibracje Spota podczas pracy Kinovy | Dodać filtr do obserwacji; opcjonalnie czekać na ustabilizowanie |

---

## Kolejność implementacji

- [ ] **Krok 1** — instalacja zależności (SB3, gymnasium)
- [ ] **Krok 2** — `button.xml` — model MuJoCo guzika ze sprężyną
- [ ] **Krok 3** — `omniboard.xml` — cały board (5 slotów)
- [ ] **Krok 4** — `CylinderInspection-v0` — klasa środowiska + reward
- [ ] **Krok 5** — test środowiska: random policy, sprawdzenie obserwacji/nagród
- [ ] **Krok 6** — `train_cylinder.py` — trening PPO, pierwsze wykresy
- [ ] **Krok 7** — `ButtonPress-v0` — klasa środowiska + reward
- [ ] **Krok 8** — `train_button.py` — trening PPO
- [ ] **Krok 9** — ewaluacja, wykresy do pracy
- [ ] **Krok 10** — `board_localizer.py` — ArUco percepcja
- [ ] **Krok 11** — `RealKinovaEnv` — wrapper Kortex API
- [ ] **Krok 12** — testy sim2real na realnym robocie

---

## Decyzje do podjęcia później

- [ ] Dokładna geometria boardu (wymiary, rozmieszczenie cylindrów) — potrzebne do `omniboard.xml`
- [ ] Siła aktywacji realnego Emergency Stop — potrzebne do kalibracji sprężyny w MuJoCo
- [ ] Czy używamy ArUco czy markerless percepcji
- [ ] Czy imitation learning (BC) jako warmstart jest wymagany przez promotora

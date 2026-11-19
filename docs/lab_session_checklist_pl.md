# Sesja w labie ALeRT: pomiary pod most 8 Hz (P8F)

**Cel sesji: dane do dostosowania mostu, bez uruchamiania polityki.** Model docelowy (P8F,
docs 6.73) działa w pętli 500/62 ≈ 8.065 Hz z limitami Web App, ze skalowaniem prędkości
w obserwacji i filtrem akcji. Most tego jeszcze nie obsługuje i nie ma backendu pod sterownik
labu `kortex_ros2_python`. Dzisiejsze pomiary rozstrzygają, czy zostajemy przy 8 Hz,
i dostarczają parametrów mostu.

**Ruch ramienia tylko w punkcie 3** (krótkie impulsy prędkości z `ros2 topic pub`), za zgodą
prowadzącego, z osobą przy E-stopie, daleko od Spota i tablicy. Wszystko inne to odczyt.

Poprzednia sesja (29.10): `docs/lab_session_checklist_2026-10-29_pl.md`, wyniki w docs 6.65.
Komendy zakładają repo w `~/Batchelor/sim2real-kinova-ppo-rl-in_robosuite`.

## Jak jest połączony lab

```
            Wi-Fi  ALeRT-4G   (192.168.50.x)
   ┌──────────────────┬────────────────────┬──────────────────────┐
 Steam Deck        laptop               PC na Spocie "max1"     Kinova Gen3
 (pulpit rqt)      (RViz, ssh,          192.168.50.10           192.168.50.9
                    ros2 bag)           ─ router zenoh :7447
                                        ─ tmux "spot_session"
                                        ─ kortex_ros2_python ───► Kortex API (TCP 10000 / UDP 10001)
```

- **W cudzym oknie tmux nie naciskać Ctrl+C** (zabija sterownik). Wyjście: **Ctrl+b d**.
- Jeśli `ros2 node list` kończy się `!rclpy.ok()`: `ros2 daemon stop; ros2 daemon start`
  (demon wystartował przed połączeniem z Wi-Fi; tak było 29.10 i 12.11).
- ✅ 12.11: ping laptop → max1 (200 pakietów co 50 ms): śr. 1.26 ms, max 4.2 ms, 0 strat.

## 0. Przed wyjściem

- [ ] Laptop, ładowarka, miarka (min. 1 m), telefon do zdjęć i nagrań.
- [ ] Hasło do `max1`, wpis `Host max1` w `~/.ssh/config` (HostName 192.168.50.10, User max1).
- [ ] Skopiować narzędzie limitów na max1 (po zgodzie, w punkcie 1):
  `scp ros2_bridge/tools/kortex_limits.py max1:/tmp/`
- [x] Katalog na nagrania: `mkdir -p logs/hw/2026-11-12`

## 1. Limity prędkości (najważniejsze: może zmienić cały plan)

Pytanie do prowadzącego, zanim cokolwiek uruchomisz:

- [ ] **Czy można podnieść w Web App limity *Cartesian Joystick* do 0.5 m/s i 50°/s**
  (oraz stawy *Angular Joystick* do ~80°/s)? Jeśli tak, wraca model P przy 20 Hz
  (100/100, ok. 10 s na zadanie, bez problemu drgań). Odpowiedź: ______
- [ ] Kto może je zmienić i czy na stałe, czy na czas testów: ______

Odczyt limitów przez Kortex API (tylko odczyt; otwiera własną sesję obok sterownika;
**za zgodą**):
```bash
# na LAPTOPIE (zwykły terminal, nie w ssh): skopiuj skrypt na max1
cd ~/Batchelor/sim2real-kinova-ppo-rl-in_robosuite
scp ros2_bridge/tools/kortex_limits.py max1@192.168.50.10:/tmp/
# zaloguj się na max1 i uruchom TAM
ssh max1@192.168.50.10
python3 -c "import kortex_api; print(kortex_api.__file__)"   # czy jest biblioteka
python3 /tmp/kortex_limits.py --ip 192.168.50.9 | tee /tmp/limity.txt
exit
# z powrotem na laptopie: zabierz wynik
scp max1@192.168.50.10:/tmp/limity.txt logs/hw/2026-11-12/
```
- [x] Twarde limity (12.11): **twist 0.5 m/s, 100°/s; stawy 80/80/80/70/70/70 °/s; przysp. 297.9 ×3, 573 ×3 °/s²**
  (= 5.2 / 10 rad/s², zgodne z modelem `HW_JOINT_ACC`)
- [x] Miękkie *Cartesian Joystick* (12.11): **0.2 m/s, 20°/s, stawy 50°/s** — dokładnie obwiednia P8/P8F;
  przyspieszeń dla tego trybu robot nie podaje
- [x] Miękkie *Angular Joystick*: stawy 50°/s, przysp. **57.3°/s² (1 rad/s²)** j1–j3, 573°/s² j4–j6.
  Trajektorie: *Angular* 25°/s (akcje `goto` jadą wolno), *Cartesian* 0.1 m/s. Plik: `logs/hw/2026-11-12/limity.txt`
- [x] `kortex_api` na max1: tak, z `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`

Tryb sterowania w czasie ruchu z drążka (ktoś porusza ramieniem Steam Deckiem, Ty patrzysz):
```bash
python3 /tmp/kortex_limits.py --ip 192.168.50.9 --watch 20
```
- [ ] Tryb podczas twistu (`CARTESIAN_JOYSTICK`?): ______
- [ ] Maksymalna zmierzona prędkość narzędzia przy pełnym wychyleniu: ______ m/s, ______ °/s

## 2. Sterownik i tematy (odczyt z laptopa)

```bash
source /opt/ros/humble/setup.bash && echo $RMW_IMPLEMENTATION   # rmw_zenoh_cpp
ros2 topic hz /joint_states
ros2 topic echo --once /joint_states
ros2 topic info -v /twist_controller/commands
ros2 run tf2_ros tf2_echo gen3_base_link end_effector_link
```
- [x] `/joint_states` publikowany? **tak** (12.11), węzeł `kinova_js_publisher`, **~1000 Hz**
  (RELIABLE, depth 10). 6 stawów ramienia + 6 stawów 2F-140 (`finger_joint` 0.699 = zamknięty).
  ⚠ W jednym pomiarze przerwy do **0.56 s** (średnia spadła 996 → 907 Hz) → powtórzyć niżej.
- [~] pole **`velocity`** jest, ale w spoczynku same 0.0 — **nierozstrzygnięte**, sprawdzić
  w ruchu (niżej). `effort` niezerowy, więc pole wypełniane z feedbacku.
- [x] zakresy kątów: **`[0, 2π)`** (j1 6.279, j4 6.261 ≈ 2π). Most rozwija kąty, OK.
  Poza z 12.11 (rad): 6.279, −1.844, −2.722, 6.261, −0.794, 1.575 — to nie jest poza startowa.
- [x] `/twist_controller/commands`: typ **`geometry_msgs/msg/TwistStamped`**, subskrybent
  **`kortex_omni_controller`** (RELIABLE, depth 1). ⚠ Jest już wydawca: **`spot_controller`**
  (ścieżka joysticka) — most / `ros2 topic pub` byłby drugim.
- [x] TF `gen3_base_link → end_effector_link`: **nie ma ramki `gen3_base_link`** → nazwy ramek niżej.
- [x] Inne: serwisy `/goto_cartesian_pose`, `/toggle_admittance`; tematy `/goto_position`,
  `/goto_tf`, `/goto_tf_offset`, `/twist_controller/gripper_vel`, kamera nadgarstka
  `/kinova_color`, `/kinova_depth_registered`; węzeł `/estop_command_subscriber`.

Do zrobienia w punkcie 2 (12.11, sam odczyt):
```bash
ros2 topic hz /twist_controller/commands          # czy spot_controller publikuje w spoczynku?
ros2 topic echo --once /twist_controller/commands # co publikuje (zera?)
ros2 topic echo --once /tf_static | grep -E "frame_id|child_frame_id"
ros2 topic echo --once /tf | grep -E "frame_id|child_frame_id"
ros2 topic info /goto_position; ros2 service type /goto_cartesian_pose   # → ros2 interface show <typ>
ros2 topic hz /joint_states --window 2000         # ~30 s bez innych poleceń: przerwy?
ros2 topic echo /joint_states --field velocity    # gdy ktoś rusza ramieniem drążkiem
```
- [x] `spot_controller` w spoczynku: **nic nie publikuje** (30 s, 12.11; Steam Deck ruszał wtedy
  tylko chwytakiem — sprawdzić ponownie, gdy joystick działa)
- [ ] ramka bazy ramienia: ______, ramka narzędzia: ______
- [x] typ `/goto_position`: `std_msgs/String` (nazwa akcji), `/goto_cartesian_pose`: `moveit_msgs/GetPositionIK` (z kodu)
- [x] przerwy w `/joint_states` (30 s w spokoju): zwykle ≤ 4 ms, sporadycznie 16–34 ms, raz **104 ms**;
  0.56 s z pierwszego pomiaru = artefakt Ctrl+C/demona. Strażnik wieku obs w moście: ~0.25 s
- [x] `velocity` w ruchu niezerowe: **tak** (18.11, bag na max1: pole velocity zgodne z różniczką pozycji, 27.3 vs 26.4 °/s); `/joint_states` 1 kHz, przerwy ≤ 5 ms przez 25 min

Co wiadomo z kodu sterownika (`kortex_omni_control.py`, `cmd_vel_cb`), do potwierdzenia:
skala **×0.5** (wiadomość 0.4 = 0.2 m/s), martwa strefa: wiadomość zerowana, gdy **wszystkie**
składowe < 0.05, **timeout 0.2 s** bez wiadomości → stop, ramka **TOOL**. Przy 8.065 Hz okres
to 0.124 s, więc jedna zgubiona wiadomość (0.248 s) zatrzymuje ramię.

## 3. Odpowiedź na skok prędkości i czy limit obejmuje API (ruch ramienia!)

**Za zgodą, z osobą przy E-stopie.** Ramię ustawione Web Appem w bezpiecznej pozie, z dala od
Spota i tablicy, z miejscem ≥ 15 cm w kierunku ruchu. Kierunek wybiera prowadzący (oś narzędzia,
ramka TOOL). Steam Deck odłożony (nie może wysyłać zer w trakcie). Admitancja: stan ______.

Terminal A, nagrywanie przez cały punkt 3:
```bash
cd ~/Batchelor/sim2real-kinova-ppo-rl-in_robosuite
ros2 bag record -o logs/hw/2026-11-12/step /twist_controller/commands /joint_states /tf
```

Terminal B: impulsy przez `--times N` przy zadanej częstotliwości `-r` (czas impulsu = N / r).
Po impulsie sterownik zatrzymuje ramię sam (timeout 0.2 s). Między impulsami odczekać 3 s.
Oś zmienić w razie potrzeby (`linear.z` = wzdłuż chwytaka):
```bash
# a) 8 Hz, 0.05 m/s przez 1 s (wiadomość 0.1 po skali x0.5)
ros2 topic pub -r 8 --times 8 /twist_controller/commands geometry_msgs/msg/TwistStamped \
  "{header: {frame_id: tool}, twist: {linear: {x: 0.1}}}"
# b) to samo w przeciwną stronę (powrót)
ros2 topic pub -r 8 --times 8 /twist_controller/commands geometry_msgs/msg/TwistStamped \
  "{header: {frame_id: tool}, twist: {linear: {x: -0.1}}}"
# c) 20 Hz, 0.05 m/s przez 1 s, tam i z powrotem (porównanie z 8 Hz)
ros2 topic pub -r 20 --times 20 /twist_controller/commands geometry_msgs/msg/TwistStamped \
  "{header: {frame_id: tool}, twist: {linear: {x: 0.1}}}"
ros2 topic pub -r 20 --times 20 /twist_controller/commands geometry_msgs/msg/TwistStamped \
  "{header: {frame_id: tool}, twist: {linear: {x: -0.1}}}"
# d) TEST LIMITU: żądane 0.3 m/s przez 0.5 s (wiadomość 0.6), tam i z powrotem.
#    Jeśli limit 0.2 m/s obejmuje API, zmierzona prędkość zatrzyma się na 0.2.
ros2 topic pub -r 8 --times 4 /twist_controller/commands geometry_msgs/msg/TwistStamped \
  "{header: {frame_id: tool}, twist: {linear: {x: 0.6}}}"
ros2 topic pub -r 8 --times 4 /twist_controller/commands geometry_msgs/msg/TwistStamped \
  "{header: {frame_id: tool}, twist: {linear: {x: -0.6}}}"
# e) obrót: 0.2 rad/s przez 1 s wokół osi narzędzia (wiadomość 0.4), tam i z powrotem
ros2 topic pub -r 8 --times 8 /twist_controller/commands geometry_msgs/msg/TwistStamped \
  "{header: {frame_id: tool}, twist: {angular: {z: 0.4}}}"
ros2 topic pub -r 8 --times 8 /twist_controller/commands geometry_msgs/msg/TwistStamped \
  "{header: {frame_id: tool}, twist: {angular: {z: -0.4}}}"
```
- [x] a)/b): **5.5–5.6 cm** przy 0.051 m/s (skala ×0.5 potwierdzona; nadmiar = ogon 0.2 s timeoutu) — 18.11, `logs/hw/2026-11-18/step`
- [x] Płynność: tętnienie prędkości stawów w plateau 2.1–2.3 % przy 8 Hz i 2.3–2.4 % przy 20 Hz — **8 Hz nie szarpie**
- [x] d): żądane 0.3 m/s → zmierzone **0.204 m/s** — limit *Cartesian Joystick* 0.2 m/s OBEJMUJE twist z API; e) 0.2 rad/s → 0.205 rad/s
- [x] Zatrzymania w trakcie impulsu: **0** (twist z laptopa przez Wi-Fi). Martwy czas komenda (odbiór na max1) → ruch ok. 30–80 ms.
  Ramię jedzie jeszcze **0.44–0.53 s po ostatniej wiadomości** (timeout 0.2 s + hamowanie) → most kończy jawnym zerem

Bag posłuży w domu do: dopasowania filtra `twist_out` przy 8 Hz (dotąd dopasowany przy 20 Hz),
pomiaru martwego czasu komenda → ruch w zegarze ściennym i rozstrzygnięcia limitu z d).

## 4. Geometria

- [ ] **Wysokość spodu podstawy Kinovy nad podłogą, Spot STOJĄCY** (zadanie na stojąco;
  12.11 Spot leżał): ______ cm (symulacja 62 cm ± 3; środek środkowego grzybka 65.5 cm)
- [x] **Przód Spota** (12.11, zdjęcia): kamera ToF **ifm O3R** (M04533, obudowa ~90×31×26 mm),
  **tył 7.5 cm, przód 10 cm przed osią podstawy, góra 4.5 cm nad grzbietem Spota**. W symulacji
  jako `lab_cam_box` z zapasem 1.5 cm (docs 6.74): P8F 0/50 kontaktów, do kamery min 9.7 cm.
  Listwa Kinecta zostaje jako bufor. Z tyłu za ramieniem: różowa skrzynka i E-stop Spota (poza
  zasięgiem pracy).
- [ ] (opcjonalnie) przesunięcie kamery w bok od osi: ______ cm; grubość płyty pod podstawą
  ramienia ponad grzbiet Spota: ______ cm
- ⚠ **NIE jechać z „Home” prosto do pozy startowej**: w symulacji (postój −0.43) liniowy ruch
  stawów Home → poza startowa wkłada palce chwytaka 0.5–2.6 cm w przód Spota (t ≈ 0.4–0.5).
  Bezpiecznie w DWÓCH krokach: Home `[0, 15, 230, 0, 55, 90]` → punkt pośredni
  **`[0, 15, 231.6, 179.9, 244.3, 180]`** (nadgarstek obraca się wysoko, min 9.9 cm) → poza
  startowa (zmienia się tylko j2 15 → 84.2; najbliżej dopiero w samej pozie, 1.0 cm).
  Najlepiej zapisać obie jako akcje Web App (`rl_via`, `rl_start`).
  **Poza spoczynkowa** (12.11): ramię oparte o karton przymocowany do Spota, stawy
  `[0, 254.4, 204.0, 358.7, 314.5, 90.3]°` (blisko „nuc”). Z niej prosto do pozy startowej:
  w symulacji OK (karton poza modelem; ruch od niego odjeżdża). Zmierzyć karton: ______
  **Z „nuc” można jechać prosto do pozy startowej** (symulacja: najbliżej dopiero w samej pozie, 1.0 cm;
  ruch oddala ramię od skrzynek za podstawą, których model nie ma). Ramię 12.11 stało w pobliżu „nuc”.
  Akcje zapisane w robocie (12.11, stopnie): Retract [0,340,214,0,310,90], Home [0,15,230,0,55,90],
  Packaging [0,330,214,0,115,270], Zero, nuc [0,254.5,212.2,0,298.8,90], Alien [0,260,355,0,0,90],
  Crate [0,0,212.2,0,7,90]. Prześwity: `training/lab_pose_clearance.py`.
- [x] (18.11: osiągnięta przez `kortex_poses.py`, najpierw j2=75, potem 84.2; prześwit ______ cm) **Poza startowa** `[0, 84.2, 231.6, 179.9, 244.3, 180]°` (stawy 1–6, Web App, powoli,
  osoba przy E-stopie): najmniejszy prześwit nadgarstka nad przodem Spota ______ cm
  (symulacja ok. 1 cm). Zdjęcie z boku. Jeśli < 2 cm albo kolizja: przerwać, poza do zmiany.
- [ ] Odległość TCP (środek palców 2F-140) od kołnierza przy zamkniętym chwytaku: ______ cm
  (scena 20.75 cm, Web App TCP Z 20 cm)

## 5. Postój Spota (z osobą od pozycjonowania)

Docelowo: **oś podstawy Kinovy 0.93 m od ściany**, Spot prostopadle do ściany, na środku tablicy.
Odpowiada to przodowi Spota ok. 0.66 m od ściany (zweryfikować z punktem 4).

| próba | oś podstawy → ściana [cm] | przesunięcie w bok od środka tablicy [cm] | skręt (na oko) |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |

- [ ] Rozrzut w X (max − min): ______ cm (trening P/P8/P8F: ±2 cm; zastąpić pomiarem)
- [ ] Jak liczona jest pozycja (ręcznie, nawigacja, znaczniki): ______

## 6. Kinematyka: kilka póz z Monitoringu

Web App → Monitoring → eksport JSON w 5–10 różnych pozach (ramię ruszane drążkiem przez
prowadzącego). Pliki do `logs/hw/2026-11-12/monitoring_*.json`. 29.10 jedna poza dała 8 mm
rozbieżności względem FK rdzenia mostu.

- [ ] liczba zapisanych póz: ______

## 7. Pytania do prowadzącego

- [ ] (punkt 1) podniesienie limitów *Cartesian Joystick*.
- [ ] Czy most może działać **na max1** (conda/venv ok. 2–3 GB, CPU)? Przez Wi-Fi zgubiona
  wiadomość > 0.2 s zatrzymuje ramię.
- [ ] Czy w `kortex_ros2_python` można dodać temat **bez martwej strefy i bez skali ×0.5**
  (np. `/twist_controller/commands_raw`) albo zmienić timeout na ~0.3 s?
- [ ] Czy `/joint_states` może zawierać prędkości (z `BaseCyclic` feedback)?
- [x] (18.11: zapisane przez `kortex_poses.py save`; `/goto_position rl_start` z `rl_via` → wynik True po 3.5 s) **Prośba: zapisać w Web App dwie akcje typu „Reach joint angles”** (gotowy tekst niżej):
  `rl_start` = `[0, 84.2, 231.6, 179.9, 244.3, 180]°` (poza startowa polityki) oraz
  `rl_via` = `[0, 15, 231.6, 179.9, 244.3, 180]°` (punkt pośredni na drodze z Home).
  Wywołanie: `ros2 topic pub --once /goto_position std_msgs/msg/String "data: rl_start"`.
  Pierwszy przejazd powoli, z osobą przy E-stopie; zmierzyć prześwit nadgarstka nad kamerą.
- [ ] Do czego służą „nuc” i „Home”? W jakiej pozie Spot chodzi? (spoczynek 12.11 = oparcie o karton)
- [ ] Zduplikowany `SendTwistCommand` w `cmd_vel_cb` (dwa wywołania na wiadomość) — zgłosić.
- [ ] `robot_state_publisher` zakomentowany w `manipulator_launch.py` → brak `gen3_base_link` w TF
  (psuje też `/goto_tf`); domyślny `gripper_type` 2f85 przy zamontowanym 2F-140.
- [ ] Czy można zobaczyć w działaniu klasyczny potok `board_mapping_launch` + `press_estop`
  (punkt odniesienia RL vs serwowanie wizyjne) i dostać jego czasy/skuteczność?
- [ ] Czy `kortex_ros2_cpp` (wersja C++ w toku) może mieć temat twistu bez martwej strefy/skali?
- [ ] Admitancja przy docisku grzybka: czy `/toggle_admittance false` jest dopuszczalne?

## 8. Notatki z sesji

- 12.11: panel z zaworami kulowymi na ścianie to stanowisko innej osoby (nie nasza tablica).
- 12.11: moduł wizyjny na nadgarstku jest w modelu robota (`bracelet_with_vision_link`).
- 12.11: kabel USB kamery ifm robi pętlę obok podstawy — przypiąć ciaśniej do płyty.
- 12.11: limity Kortex: `kortex_limits.py` wymaga `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`;
  limity/tryb w serwisie ControlConfig (nie Base). Wynik limitów: ______ (`logs/hw/2026-11-12/limity.txt`).
- 12.11: render póz w symulacji: `logs/hw/2026-11-12/poses_sim.png` (`training/render_lab_poses.py`).
- 12.11: Home jako poza startowa — rozważona, odłożona (wymaga dotrenowania; wraca, jeśli prześwit
  w `rl_start` < 2 cm albo kontakty na starcie się utrzymają; wtedy start losowany z obu póz).

### Tekst prośby do labu (EN)

> Hi, could you please save two "Reach joint angles" actions in the Web App for my RL tests?
> `rl_start`: [0, 84.2, 231.6, 179.9, 244.3, 180] deg and `rl_via`: [0, 15, 231.6, 179.9, 244.3, 180] deg.
> From the rest pose (arm on the cardboard box) or from `nuc` the arm can go straight to `rl_start`;
> from `Home` it should go through `rl_via` first — in simulation the straight Home → rl_start joint
> move dips the gripper fingers into the front of Spot. In `rl_start` the wrist is only ~1 cm above
> the front camera in my model, so the first run should be slow, with someone at the E-stop.
> I would trigger them via `/goto_position`. Thank you!
- 18.11: sesja sama, Spot leżący, stos uruchomiony ręcznie (`manipulator_launch.py`, `kinova_vision`) w sesji tmux `rl_arm`.
  Ruch przez `ros2_bridge/tools/kortex_poses.py` (Kortex API z max1, bez Steam Decka). Przejazdy: spoczynek → j2=75
  (18 s, 10 °/s) → `rl_start` (2 s) → `rl_via` ↔ `rl_start` (akcje, 3.2 s, szczyt 26–28 °/s = limit Angular
  Trajectory) → Home (7.4 s, tylko do pomiaru FK; ten sam odcinek co Home → `rl_via` w przeciwną stronę). FK symulacji vs Kortex: 4–9 mm (spoczynek, Home; różnica TCP 20 vs 20.75 cm).
  Test skoku (punkt 3) powtórzony z nagrywaniem (`step`, w Home, oś narzędzia z). Ruch Home ↔ `rl_via` w danych gładki
  (stałe ~25 °/s na j4/j5), wizualnie niepokojący: obrót nadgarstka j4 180°, j5 170°. Trasa: `nuc` → Home → `rl_via` → `rl_start`; powrót tą samą drogą: `rl_start` → `rl_via` → Home → `nuc` (sym.: Home ↔ `nuc` min 9.9 cm).
  Bag: `logs/hw/2026-11-18/session`, pozy: `logs/hw/2026-11-18/poses.jsonl`.

# Sesja w labie ALeRT: rekonesans przed mostem `arm_d`

**Cel sesji: tylko obserwacja i pomiary, bez ruchu ramienia z mostu.** Lab steruje Kinovą
własnym sterownikiem `kortex_ros2_python` (RRL-ALeRT, Kortex Python API), a nie `ros2_kortex`.
Most nie ma jeszcze backendu pod ten sterownik. Dzisiejsze odpowiedzi wyznaczą, jak go napisać.

Komendy zakładają repo w `~/Batchelor/sim2real-kinova-ppo-rl-in_robosuite`.

## Jak jest połączony lab

```
            Wi-Fi  ALeRT-4G   (192.168.50.x)
   ┌──────────────────┬────────────────────┬──────────────────────┐
 Steam Deck        laptop               PC na Spocie "max1"     Kinova Gen3
 (pulpit rqt)      (RViz, ssh,          192.168.50.10           192.168.50.9
                    ew. most)           ─ router zenoh :7447
                                        ─ tmux "spot_session"
                                        ─ tmux_api_server
                                        ─ sterownik Spota
                                        ─ kortex_ros2_python ───► Kortex API (TCP 10000 / UDP 10001)
```

- **max1** uruchamia wszystkie węzły. Każdy przycisk pulpitu na Steam Decku to okno w sesji
  tmux `spot_session` (start/stop przez `tmux_api_server`).
- **ROS 2 przez zenoh:** router na max1, laptop jako klient (`~/.bashrc`:
  `RMW_IMPLEMENTATION=rmw_zenoh_cpp`, `ZENOH_CONFIG_OVERRIDE` → `tcp/192.168.50.10:7447`).
- README pulpitu podaje `192.168.1.100` / `192.168.1.10`. To stare adresy; obowiązują `.50.10` i `.50.9`.

## 0. Przed wyjściem

- [ ] Laptop, ładowarka, telefon do nagrywania.
- [ ] Hasło do `max1` (od prowadzącego) i wpis w `~/.ssh/config`:
  ```
  Host max1
      HostName 192.168.50.10
      User max1
  ```
- [ ] Opcjonalnie: `sudo apt install ros-humble-octomap-rviz-plugins` (RViz z konfiguracją nawigacji).

## 1. Sieć i ROS

1. Podłączyć laptop do Wi-Fi **ALeRT-4G**.
2. Nowy terminal (ze zwykłym `~/.bashrc`, czyli z zenoh, **bez** domowego `rmw_fastrtps_cpp`):
   ```bash
   source /opt/ros/humble/setup.bash
   echo $RMW_IMPLEMENTATION          # ma być rmw_zenoh_cpp
   ping -c 3 192.168.50.10           # max1
   ping -c 3 192.168.50.9            # Kinova
   ros2 node list                    # czy widać węzły robota
   ```
3. **Opóźnienie Wi-Fi** (wchodzi w martwy czas pętli, jeśli most działa na laptopie):
   ```bash
   ping -c 200 -i 0.05 192.168.50.10 | tail -2
   ```
rtt min/avg/max/mdev = 0.844/1.256/4.220/0.388 ms

## 2. Odczyt tematów z laptopa (nic nie wysyła)

```bash
ros2 topic list | grep -iE "joint|twist|kinova|goto|gripper"
ros2 topic hz /joint_states
ros2 topic echo --once /joint_states
ros2 topic info -v /twist_controller/commands
ros2 topic info -v /joint_states
ros2 run tf2_ros tf2_echo gen3_base_link end_effector_link
ros2 service list | grep -iE "admittance|goto|cartesian"
```

- [ ] `/joint_states`, częstotliwość: ______ Hz
- [ ] `/joint_states` ma pole **`velocity`**? tak / nie (obserwacja polityki zawiera prędkości stawów)
- [ ] nazwy stawów ramienia (`joint_1`…`joint_6`?): ______
- [ ] zakres j1, j4, j6 (`[0, 2π)` czy `[−π, π]`?): ______
- [ ] kto publikuje `/joint_states` (węzeł): ______
- [ ] `/twist_controller/commands`: typ (`TwistStamped`?) ______, publikujący ______, subskrybujący ______
- [ ] TF `gen3_base_link → end_effector_link` istnieje? tak / nie; kto go publikuje: ______
- [ ] usługi `/toggle_admittance`, `/goto_cartesian_pose` widoczne? ______

## 3. max1 przez ssh i tmux (tylko patrzeć)

```bash
ssh max1
tmux ls
tmux attach -t spot_session
#   Ctrl+b w   lista okien (strzałki + Enter)
#   Ctrl+b d   ODŁĄCZENIE: wyjście bez zatrzymywania czegokolwiek
```

**W cudzym oknie tmux nie naciskać Ctrl+C**, bo to zabija proces (sterownik Spota albo Kinovy).
Wyjście zawsze przez **Ctrl+b d**.

- [ ] okno z Kinovą, nazwa: ______, komenda: ______
- [ ] skąd startuje `kortex_ros2_python` (ścieżka workspace na max1): ______
- [ ] wersja repo na max1 (`git log --oneline -1` w jego katalogu): ______ (u nas `7ebdd34`)
- [ ] Python i ROS na max1 (`python3 --version`, `echo $ROS_DISTRO`): ______
- [ ] wolne miejsce i CPU (`df -h ~`, `nproc`, `free -h`), na środowisko mostu ok. 2–3 GB: ______

## 4. Steam Deck w trybie Kinovy (obserwacja)

Na laptopie:
```bash
ros2 topic echo /twist_controller/commands
```
Prowadzący przełącza Steam Deck na Kinovę i porusza drążkiem.

- [ ] co przychodzi przy ruchu drążka (zakres wartości): ______
- [ ] **czy przy puszczonym drążku lecą ciągle zera?** tak / nie (jeśli tak, Steam Deck zatrzymywałby most)
- [ ] czy po przełączeniu na Spota wiadomości przestają przychodzić? tak / nie
- [ ] inne tematy w trybie Kinovy (`/kinova_joy`, `/twist_controller/gripper_vel`): ______

Opcjonalnie, nagranie do analizy w domu (martwy czas odpowiedzi ramienia na drążek):
```bash
cd ~/Batchelor/sim2real-kinova-ppo-rl-in_robosuite && mkdir -p logs/hw
ros2 bag record -o logs/hw/steamdeck_kinova /twist_controller/commands /joint_states /tf
```

## 5. Web App Kinovy (`http://192.168.50.9`, jeśli dostępna z Wi-Fi)

- [ ] wersja firmware: ______
- [ ] liczba stawów (ma być **6**): ______
- [ ] chwytak (2F‑85?): 2f 140______
- [ ] **TCP narzędzia, przesunięcie Z** (most: `--tool-tcp-z`, domyślnie 0.1448 m): ______
  (w `kortex_ros2_python` jest `tool_tip` +0.17 m i offset 0.203 m — sprawdzić, który obowiązuje)
- [ ] zapisane akcje (Actions), np. „Home”, „nuc”: ______
- [ ] strefy ochronne i limity (zdjęcie ekranu)
- [ ] domyślny tryb admitancji po starcie: ______

## 6. Pytania do prowadzącego

- [ ] Czy most może działać **na max1** (osobne środowisko conda/venv, ok. 2–3 GB, inferencja na CPU)?
  Na laptopie pętla idzie przez Wi-Fi, a budżet polityki to ok. 1 krok (50 ms).
- [ ] Czy można dodać do `kortex_ros2_python` **osobny temat bez martwej strefy**
  (np. `/twist_controller/commands_raw`)? Obecnie twist ze wszystkimi składowymi < 0.05
  jest zerowany, a to dokładnie końcowy dojazd i docisk do grzybka.
- [ ] Czy można publikować **prędkości stawów** w `/joint_states` (z `BaseCyclic` feedback)?
- [ ] Jak bezpiecznie przełączać Steam Deck ↔ most (żeby drążek nie wysyłał zer w trakcie biegu)?
- [ ] Czy można zapisać w Web App pozę startową `arm_d` jako akcję
  (`0, 1.4692, −2.2410, 3.1403, −2.0200, 3.1416` rad) do bazowania przez `/goto_position`?
- [ ] Admitancja przy docisku: czy wyłączenie (`/toggle_admittance false`) jest dopuszczalne?
- [ ] Gdzie jest realna tablica z grzybkami i czy można zmierzyć siłę wciśnięcia (waga kuchenna)?
- [ ] Czy można **podłączyć laptop kablem do sieci ramienia** (switch na Spocie albo port max1)
  i na czas testu **zatrzymać okno Kinovy w tmux**, żeby użyć `ros2_kortex` (backend `kortex` mostu
  jest gotowy)? Przez Wi-Fi to odpada: pętla `ros2_control` ok. 1 kHz, zabezpieczenia po stronie
  laptopa giną razem z łącznością, martwy czas z Wi-Fi. Do kabla: gdzie się wpiąć i jaki adres
  ustawić laptopowi: ______

## 7. Notatki z sesji

## Dodatek: tor `ros2_kortex`

Most ma gotowy backend `kortex` dla upstreamowego `ros2_kortex` (sprawdzony na atrapie i na
`kortex_bringup` z fake hardware). **W labie nie jest używany.** Komendy: README, sekcja
„Running on the robot”.

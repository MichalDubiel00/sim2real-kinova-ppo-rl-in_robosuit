# Modele: manifest kontraktów scen

**Checkpoint bez kontraktu sceny jest pułapką, nie wartością.** Wczytuje się **bez
błędu** w zmienionej scenie i jedynym objawem jest niski wynik — tak straciliśmy
pomiary, zanim `compare_models.py` dostał `--base-x-offset` (docs 6.42). Dlatego
każda pozycja poniżej podaje flagi, przy których jej liczba została zmierzona,
**n** i **horyzont**. Liczby bez tych trzech rzeczy nie znaczą nic.

Pliki są w `logs/` (poza gitem) i w kopii na `gs://sim2real_robosuite/backup/`.
Wartości w kolumnie „scena" wyciągnięte wprost z configów, nie z pamięci.

## MODEL NA SPRZĘT

### `arm_d/checkpoints/ppo_board_98123552_steps.zip`
- config `training/configs/arm_d.yaml` = `arm_sb` + 8M z martwym czasem
  `action_delay_steps: [0, 2]` losowanym per epizod; h=2000, n=100, ziarno 1234,
  tryb `scale` (docs 6.55)
- **d=0: 98/100, kolizje 3.46 (mediana 0.0) | d=1: 99/100, kolizje 4.90 |
  d=2: 98/100, kolizje 18.74** (d = opóźnienie akcji w krokach po 50 ms)
- scena **identyczna jak `arm_sb` i e6ab** (patrz niżej); jedyna różnica w ocenie
  to flaga `--action-delay`
- kopia: `gs://sim2real_robosuite/arms/arm_d/` i `backup/decyzyjne/`

```bash
POZA="0.0,1.4692,-2.2410,3.1403,-2.0200,3.1416"
python training/compare_models.py "arm_d=logs/board_6dof/arm_d/checkpoints/ppo_board_98123552_steps.zip" \
  --robot Kinova3_6DOF --episodes 100 --seed 1234 --horizon 2000 \
  --kinova-limits --acc-limits --acc-mode scale --action-delay 0 \
  --estop-config training/configs/ppo_board_6dof_estop_hw.yaml \
  --spot-body --spot-front-cam --base-x-offset=-0.34 --init-qpos="$POZA"
```

**Rezerwa: `arm_sb/checkpoints/ppo_board_90123552_steps.zip`** — d=0: 100/100,
kolizje 6.98; d=2: 93/100, kolizje 30.68 (n=100, h=2000, scale). Wraca tylko,
jeśli martwy czas zmierzony na sprzęcie w zegarze ściennym jest trwale < 50 ms
bez ogona (docs 6.55).

## Poprzedni model na sprzęt (do 2026-10-09)

### `spot_e6ab/checkpoints/ppo_board_86123552_steps.zip`
- config `ppo_board_6dof_spot_e6ab.yaml`, h=2000, n=100
- **100/100 [96.3–100 %], 15.2 s, kolizje 7.0 (mediana 6), misuse 0.0,
  każdy przycisk 100/100, 0 zacięć** (docs 6.50)
- ⚠ **Przy wierniejszym trybie `--acc-mode scale` (tak przycina firmware): 40/40,
  14.6 s, kolizje 11.9 (mediana 6.5), n=40.** Na sprzęt cytować **11.9**, nie 7.0
- scena: korpus Spota + kamera, `base_x_offset -0.34`,
  `init_qpos [0.0, 1.4692, -2.2410, 3.1403, -2.0200, 3.1416]`,
  prędkości [1.39×3, 1.22×3] tryb scale, przyspieszenia [5.2×3, 10.0×3],
  twist [0.5, 0.8727], E-stop (skok 6 mm, 20 N, zatrzask), `activation_depth 0.0059`

```bash
POZA="0.0,1.4692,-2.2410,3.1403,-2.0200,3.1416"
python training/compare_models.py "e6ab=logs/board_6dof/spot_e6ab/checkpoints/ppo_board_86123552_steps.zip" \
  --robot Kinova3_6DOF --episodes 100 --seed 1234 --horizon 2000 \
  --kinova-limits --acc-limits --acc-mode scale \
  --estop-config training/configs/ppo_board_6dof_estop_hw.yaml \
  --spot-body --spot-front-cam --base-x-offset=-0.34 --init-qpos="$POZA"
```

## Poprzednie modele — każdy w INNEJ scenie

Wolno porównywać tylko w obrębie jednej sceny. Zestawianie liczb między wierszami
tej tabeli jest błędem (popełnionym już dwa razy, docs 6.42 i 6.44).

| model | scena | wynik |
|---|---|---|
| `spot_e6a/.../78123552` | jak e6ab | 98/100 [93.0–99.4], 15.6 s, kolizje 8.7, misuse 0.7, n=100 |
| `spot_e5/.../74123552` | jak e6ab | 38/40 (95 %), 17.5 s, kolizje 20.9 (med. 9), misuse 0.2, n=40 |
| `spot_e2b/.../50223552` | jak e6ab, ale **BEZ limitu przyspieszeń** | 100/100 [96.3–100], 7.4 s, kolizje 0.2, n=100 |
| `kinova_ft_lr2/.../28023552` | **BEZ korpusu Spota**, `base_x -0.10`, poza domyślna, bez limitu przyspieszeń | 92/100 [85–96], 23.9 s, kolizje 2.3 (med. 1), n=100 |
| `sweep/ft_sc1_v23_long/ppo_board_final.zip` | **INNE ZADANIE** — patrz niżej | 76/100 @h=1500, 28.3 s, kolizje 7.9 |

### ⚠ `ft_sc1_v23_long` mierzy inne zadanie, nie tylko inną scenę

Jego config ma `activation_depth 0.012` i **brak profilu E-stopu** — czyli przycisk
liniowy 500 N/m z progiem 12 mm, model sprzed poprawki wg karty katalogowej
(docs 6.33: wierny przycisk ma skok 5 mm i 25 N). Dodatkowo `horizon: 1000`
w configu, a wynik 76/100 raportowany przy h=1500. Ten model **nie jest**
porównywalny z żadnym wierszem powyżej żadną flagą — służy tylko jako zapis etapu.

## Co jest w kopii na buckecie

- `backup/decyzyjne/` — modele z tego manifestu (arm_d i arm_sb dodane 2026-10-15)
- `backup/best_model/` i `backup/final/` — po jednym na bieg (45 + 29 plików)
- `backup/eval/` — CSV-ki z ewaluacji (surowe dane za liczbami w dokumentacji)
- `backup/tensorboard/` — krzywe treningowe

**Nie ma** 2319 pośrednich checkpointów (5.9 GB) — to rusztowanie przebiegu,
odtwarzalne i bez wartości dowodowej.

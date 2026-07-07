# Dokumentacja projektu: Uczenie ze wzmocnieniem manipulatora Kinova Gen3 — zadanie ButtonBoard (omniboard) w pipeline sim2real

Stan na: 2026-07-07. Dokument roboczy pod pracę inżynierską — zawiera opis środowiska,
decyzje projektowe z uzasadnieniami, przebieg eksperymentów oraz napotkane problemy
techniczne wraz z rozwiązaniami.

---

## 1. Cel projektu

Wytrenowanie w symulacji (robosuite/MuJoCo) polityki sterowania ramieniem Kinova Gen3
zamontowanym na robocie kroczącym Boston Dynamics Spot, realizującej zadanie z zawodów
RoboCup Rescue: **wciśnięcie 5 przycisków Emergency Stop rozmieszczonych na panelu
"omniboard"** (krzyż ze skośnymi ścianami), z założeniem późniejszego przeniesienia
polityki na prawdziwego robota (sim2real).

Kluczowe założenie architektury sim2real: polityka RL widzi **wyłącznie wielkości
odtwarzalne na prawdziwym sprzęcie** (rozdz. 5).

## 2. Stack technologiczny

| Komponent | Wersja | Rola |
|---|---|---|
| robosuite | 1.5.1 | framework środowisk manipulacyjnych nad MuJoCo |
| MuJoCo | 3.3.7 | silnik fizyki |
| stable-baselines3 | 2.3.2 | implementacja PPO |
| gymnasium | 0.29.1 | interfejs środowiska |
| PyTorch | 2.12 (CUDA) | sieci neuronowe |
| numpy | 1.26.4 | |
| Python | 3.10 (conda env `robosuite`) | |

Sprzęt treningowy: laptop Legion, i7-9750H (6 rdzeni / 12 wątków), RTX 2060, 15 GB RAM.

## 3. Struktura projektu

```
robosuite/
├── envs/
│   ├── button_board.py          # ButtonBoardEnv — środowisko zadania (główne)
│   ├── button_press.py          # ButtonPressEnv — 1 przycisk na stole (etap pośredni)
│   └── cylinder_inspection.py   # CylinderInspectionEnv (inne zadanie)
├── objects/
│   ├── button_object.py         # ButtonObject — przycisk E-Stop (MujocoXMLObject)
│   └── custom_objects.py        # OmniBoardObject, linearboardObject
├── my_assets/objects/
│   ├── xml/button.xml, omniboard.xml, ...
│   └── meshes/Omniboard.stl, "Emergency stop.stl", ...
├── training/
│   ├── train_board.py           # trening PPO (argparse + YAML, SubprocVecEnv)
│   ├── watch_board.py           # wizualizacja polityki (okno na żywo / --record mp4)
│   └── configs/ppo_board.yaml   # hiperparametry
├── logs/                        # checkpointy, eval, tensorboard, stdout
└── docs/dokumentacja_projektu.md
```

## 4. Środowisko symulacyjne `ButtonBoardEnv`

### 4.1. Scena

- Robot: **Kinova3 + gripper Robotiq 85**, baza podniesiona na wysokość
  `KINOVA_BASE_Z = 0.65 m` (korpus Spota ~0.5 m + mocowanie ~0.15 m). Robot stoi
  w origin, patrzy w +X.
- Ściana w `x = WALL_X = 0.50 m` (wizualna), na niej **omniboard**.
- Kontroler: composite **BASIC** robosuite → efektywnie OSC_POSE; akcja 7D:
  6 delt pozy końcówki (pozycja + orientacja) + 1 DOF chwytaka, zakres [-1, 1].
- Częstotliwość sterowania: 20 Hz, horyzont epizodu: 1000 kroków (50 s).

### 4.2. Omniboard — geometria (odtworzona z mesha i zdjęcia z RoboCup)

Mesh `Omniboard.stl`: krzyż **320×320×80 mm** (ramiona szerokie 80 mm), bryła
**zwęża się** od dużej ściany tylnej do małej frontowej; ramiona kończą się
**skosami 45°**. Montaż: duża ściana przylega do muru, mała frontowa skierowana
do robota.

Rozmieszczenie przycisków (klucz do zadania — na podstawie zdjęcia z RoboCup):
- **btn0**: mała ściana frontowa, oś wciskania prosto w mur (world +X),
- **btn1–btn4** (góra/prawo/dół/lewo): na **skośnych ścianach 45°** ramion — każdy
  wciskany z innego kierunku (stąd nazwa "omni"). Środki skosów: 120 mm od środka
  krzyża, 40 mm przed murem.

Implementacja: tabela `BUTTON_FACES = [(offset_środka_ściany, normalna_ściany), ...]`
(układ świata); kwaterniony przycisków liczone z normalnych
(`_face_quat`: konstrukcja bazy ortonormalnej z osią Z = normalna).

Kolizja boardu: **ten sam mesh** (MuJoCo używa otoczki wypukłej — płaszczyzny skosów
i frontu leżą na otoczce, więc przyciski stykają się z nimi dokładnie).

### 4.3. Model fizyczny przycisku (`button.xml`)

- Obudowa (statyczna): żółty box 66×66×40 mm.
- Grzybek (ruchomy): czerwony walec Ø44×24 mm na szyjce, na **przegubie
  pryzmatycznym** wzdłuż osi przycisku:
  - skok: 15 mm (`range="0 0.015"`),
  - sprężyna: k = 500 N/m, tłumienie 20, tarcie 1,
  - **próg aktywacji: 12 mm** wciśnięcia (`ACTIVATION_DEPTH = 0.012`),
  - siła pełnego wciśnięcia: F = k·x = 500·0.012 = **6 N** (realistyczna dla E-Stopa).
- Maski kolizji: obudowa `contype=2/conaffinity=1`, grzybek `4/1` — **nie kolidują
  ze sobą** (grzybek konstrukcyjnie wsuwa się w obudowę), oba kolidują z robotem (1/1).

### 4.4. Przestrzeń obserwacji (wersja sim2real, od 2026-07-06)

Wszystkie obserwacje są **odtwarzalne na prawdziwym robocie** — tabela mapowania:

| Obserwacja | Wymiar | Źródło w symulacji | Źródło na sprzęcie |
|---|---|---|---|
| propriocepcja (kąty, prędkości, poza EEF, stan chwytaka) | ~40 | MuJoCo | enkodery + kinematyka Kinovy |
| `target_relative_pos` | 3 | prawda + szum percepcji | percepcja (poza boardu) − kinematyka |
| `target_distance` | 1 | norma powyższego | j.w. |
| `target_onehot` | 5 | maszyna stanów env | maszyna stanów nadzorcy |
| `target_press_axis` | 3 | normalna ściany + szum | z estymowanej pozy boardu |
| `eef_align_with_target` | 1 | kąt oś EEF ↔ oś celu | kinematyka + poza boardu |

**Usunięte świadomie**: `all_depressions` (wciśnięcia sprężyn wszystkich przycisków) —
przyciski na prawdziwym boardzie **nie są podłączone elektrycznie**, więc ta wielkość
jest nieobserwowalna. Detekcja wciśnięcia na sprzęcie: **siła na końcówce (estymowana
z momentów w stawach Kinovy) + głębokość dopchnięcia z kinematyki** (≥12 mm wzdłuż osi
od kontaktu) — dokładnie ten sam warunek co w symulacji, tylko mierzony po stronie robota.

### 4.5. Randomizacja domeny (domain randomization)

| Parametr | Wartość | Co symuluje |
|---|---|---|
| `board_pos_noise = 0.05` | ±5 cm YZ, per epizod | niepewność pozycji boardu / ustawienia Spota |
| `random_order = True` | losowa kolejność 5 przycisków | goal-conditioning — dowolna komenda operatora |
| `percep_pos_bias = 0.005` | ±5 mm, stały per epizod | błąd estymacji pozy boardu (np. AprilTag) |
| `percep_pos_noise = 0.002` | σ = 2 mm, per krok | szum pomiaru percepcji |
| `percep_axis_noise = 0.02` | perturbacja osi, per epizod | błąd orientacji estymowanej pozy |

Błąd percepcji ma strukturę **bias (per epizod) + szum (per krok)** — to wiernie oddaje
charakter błędu systemu wizyjnego (stała pomyłka estymacji pozy + drganie pomiaru).
Nagroda (rozdz. 4.6) liczona jest z **prawdziwego** stanu symulacji — nagroda istnieje
tylko w treningu, więc może być uprzywilejowana; szum dotyczy wyłącznie obserwacji polityki.

### 4.6. Funkcja nagrody — wersja 1 (z diagnozą wady)

```
r = exp(-8·d) + 0.5·exp(-3·θ) + 5·[wciśnięcie celu] + 20·[komplet] − 0.5·[kolizja z płytą]
r_norm = r / 46.5
```
gdzie d — odległość EEF od celu, θ — kąt osi EEF do osi wciskania celu.

**Zdiagnozowana wada (2026-07-07, po ~2.7M kroków treningu):** nagroda za *przebywanie*
blisko celu (~1.3 pkt/krok) w połączeniu z przeskokiem celu po wciśnięciu tworzy
**lokalne optimum "wiszenia"**: wciśnięcie daje +5 jednorazowo, ale przelot do następnego
przycisku (~0.2 m, ~40 kroków przy obniżonym approach) kosztuje ~35 pkt. Wciskanie jest
netto nieopłacalne → polityka uczy się dolatywać do pierwszego celu i wisieć przy nim
(obserwowane zachowanie: szybki dolot, potem chaotyczne ruchy z rzadkimi przypadkowymi
wciśnięciami; eval reward w plateau ~13–15 od 2.35M kroków).

**Poprawka — wersja 2 (wdrożona 2026-07-07):**

```
r = 10·(d_prev − d_curr)                     # potential-based shaping (postęp, nie obecność)
  + 0.05·exp(-8·d) + 0.05·exp(-3·θ)          # śladowy gradient gęsty
  + 25·[wciśnięcie celu] + 100·[komplet]     # bonusy dominujące
  − 0.5·[kolizja z płytą] − 0.01·‖a‖²        # kary
r_norm = r / 225 ;  epizod kończy się po sukcesie (bonus kompletu = nagroda terminalna)
```

Własności: wiszenie w miejscu daje ~0 (szaping płaci tylko za realny postęp; przeskok
celu po wciśnięciu nie generuje kary — potencjał resetowany przy zmianie celu),
wciskanie jest jednoznacznie opłacalne (+25 ≫ koszt przelotu ~0), terminacja po
sukcesie usuwa dochód z "dożywania" epizodu, kara akcji tłumi szarpanie (istotne też
dla sprzętu). Skala nagród nieporównywalna z v1: epizod losowej polityki ≈ 0÷−0.2,
pełny sukces ≈ +1.0.

### 4.7. Kryterium sukcesu i maszyna stanów

Zadanie sekwencyjne: env utrzymuje `_order` (permutacja przy `random_order=True`)
i `_current_idx`; cel bieżący eksponowany przez `target_onehot`. Wciśnięcie celu
(≥12 mm) przesuwa wskaźnik. Sukces = wszystkie 5 wciśnięte. Na sprzęcie identyczną
rolę pełni węzeł nadzorcy (rozdz. 5).

## 5. Koncepcja przeniesienia na sprzęt (sim2real)

```
[kamera] → percepcja: poza 6D omniboarda        (AprilTag — rekomendowane na start,
                                                 YOLO+depth — rozszerzenie)
        → pozycje i osie 5 przycisków z geometrii CAD (BUTTON_FACES)
        → [polityka PPO]: obs = propriocepcja + cel względny + oś + one-hot
        → delty OSC → kontroler Kinovy
[detekcja wciśnięcia]: siła z momentów stawów + głębokość z kinematyki (≥12 mm)
        → maszyna stanów nadzorcy przełącza cel (one-hot) na następny
```

Uzasadnienie wyboru percepcji:
- **AprilTag/ArUco**: jedna estymacja pozy 6D załatwia wszystkie przyciski (sztywna
  geometria), znane błędy (±mm/±1°) — dokładnie te, które symulujemy szumem percepcji;
  pipeline end-to-end osiągalny szybko.
- **YOLO + depth**: bez markerów (realistyczne zawodniczo), ale wymaga datasetu,
  kalibracji kamera–robot, i nadal dopasowania modelu boardu dla orientacji skosów.
  Sensowne jako rozdział porównawczy.
- End-to-end RL z obrazu: odrzucone (koszt treningu i kruchość transferu).

Przyciski niepodłączone elektrycznie → detekcja wciśnięcia wyłącznie po stronie robota
(siła+kinematyka). Dodatkowa weryfikacja wizyjna możliwa: prawdziwy E-Stop **zatrzaskuje
się** po wciśnięciu (grzybek zostaje ~10 mm niżej) — symulacja modeluje sprężynę powrotną,
co dla polityki jest bez różnicy (liczy się moment aktywacji).

## 6. Trening

### 6.1. Hiperparametry PPO (`ppo_board.yaml`)

| Parametr | Wartość |
|---|---|
| polityka | MlpPolicy, net_arch [256, 256], tanh |
| learning_rate | 3e-4 |
| n_steps | 2048 (per env) |
| batch_size | 64 |
| n_epochs | 10 |
| gamma / gae_lambda | 0.99 / 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.005 |
| n_envs | 8 (SubprocVecEnv) |
| total_timesteps | 3M |
| horizon | 1000 |

### 6.2. Wydajność obliczeniowa (pomiary własne)

- Trening używa GPU (`Using cuda device`), ale **wąskim gardłem jest symulacja MuJoCo
  na CPU** (GPU ~1% przy MLP 256×256) — skalowanie przez liczbę równoległych środowisk.
- 4 envy: ~230–250 FPS chwilowo, ~130 FPS w długiej sesji.
- 8 envów: 346 FPS na starcie → **mediana 152 FPS** w wielogodzinnej sesji —
  **thermal throttling** laptopowego i7-9750H przy ciągłym obciążeniu wszystkich rdzeni.
- Praktyczny limit envów na 6C/12T: ~10 (MuJoCo = czyste obliczenia, hyperthreading
  daje ~20–30%; przy >10 workerach walka o rdzenie niweluje zysk). RAM: ~0.8 GB/proces.
- Realny czas 3M kroków na tej maszynie: **~5.5 h** (nie ~2 h z naiwnego przeliczenia
  szczytowego FPS — wniosek: szacować po ustabilizowanym FPS, nie po pierwszej iteracji).

### 6.3. Przebieg eksperymentów

| Run | Geometria | Obserwacje | Wynik | Wniosek |
|---|---|---|---|---|
| v1 (`logs/board_v1_flat_layout`) | **błędna**: 5 przycisków płasko na froncie | uprzywilejowane (all_depressions) | eval ~10.4–11.2 @ 2.5–3M | układ niezgodny z RoboCup — geometria odtworzona ze zdjęcia, restart |
| v2 (`logs/board_v2_privileged_obs`) | poprawna (krzyż + skosy 45°) | uprzywilejowane | eval 4.58 @ 100k (przerwany) | przejście na obs sim2real |
| v3 (`logs/board`, reward v1) | poprawna | **sim2real** (szum percepcji, bez all_depressions) | eval: 0.77@50k → 6.83@200k → **plateau 13–15 od 2.35M**; ~1/5 przycisków/epizod, ruchy chaotyczne | **wada rewardu — lokalne optimum wiszenia** (rozdz. 4.6); wymaga rewardu v2 |
| v4 (`logs/board`, reward v2) | poprawna | sim2real | **pierwsze pełne sukcesy**: 0% → ~50–70% @ 2.8–3M; best eval 0.739 @ 2.8M; krzywa nadal rośnie przy 3M | reward v2 usunął lokalne optimum; polityka niedotrenowana — kontynuować |

**Wyniki v4 — finalne (6M kroków = 3M od zera + 3M wznowienia):** krzywe uczenia
0–6M — `docs/img/v4_eval_curve.png` (odtworzone z eventów TensorBoard, run
`ppo_board_5`). Przebieg:
- 0–1.5M: nauka dolotu i celowania (reward rośnie od −0.16, zero sukcesów),
- ~1.55M: pierwsze pełne sukcesy (5/5 przycisków z terminacją),
- 3M: ~50–70% sukcesów, best eval 0.739 @ 2.8M; krzywa wciąż rośnie → wznowienie,
- 6M: **92% sukcesów średnio w ostatnich 10 ewaluacjach** (kilka ewaluacji 100%),
  best eval 0.961 @ 4.9M; średnia długość epizodu spadła z 1000 (horyzont) do ~450–550.

Test najlepszego modelu na 5 nagranych epizodach (`v4_final_6M.mp4`): **5/5 sukcesów**,
epizody 271–577 kroków (14–29 s), rewardy 0.93–0.98. Dla porównania model @ 2.8M
miał 3/5 i ~790 kroków/udany epizod — kontynuacja treningu dała i skuteczność,
i ~2× szybsze wykonanie.

Incydent: pierwsza próba wznowienia padła po ~98k kroków z `CUDA error: unspecified
launch failure` (niestabilność laptopowego GPU po ~8 h ciągłego obciążenia; nie błąd
kodu). Ponowne wznowienie z checkpointu 3M przeszło bez problemów — checkpointowanie
co 100k kroków okazało się kluczowym zabezpieczeniem.

Uwaga metodologiczna: wznowienie nadpisuje `evaluations.npz` (EvalCallback zaczyna
plik od nowa) — pełną historię ewaluacji trzymają eventy TensorBoard; przy wznowieniu
z `reset_num_timesteps=False` SB3 dopisuje do tego samego runu TB, więc krzywa jest
ciągła.

**Analiza porażek — 100 epizodów best_model @ 6M** (`logs/board/eval100.log`):
sukcesy 89/100, średni czas sukcesu 528 kroków (26 s). Porażki: 9× częściowy postęp
(2–4/5, zabrakło horyzontu — niegroźne), 1× utknięcie, 1× 0/5. **Kluczowe odkrycie:
96/100 epizodów miało kontakt ramienia z płytą** (śr. 17 kroków ≈ 0.85 s/epizod) —
kara kolizji −0.5 po normalizacji (0.002/krok) była pomijalna wobec bonusów i polityka
nauczyła się ją ignorować. W symulacji bez znaczenia, na sprzęcie niedopuszczalne
(szorowanie chwytakiem po panelu).

**Reward v2.1 (fine-tuning od checkpointu 6M, +2M kroków):**
- kara kolizji ×10: −5.0 (COLLISION_COST),
- nowa kara za ZMIANĘ akcji: −0.02·‖aₜ−aₜ₋₁‖² (ACTION_RATE_COST) — karze szarpanie,
  nie prędkość; wygładzenie trajektorii istotne dla sprzętu (przekładnie, odrzucane
  komendy przy skokowych deltach OSC).
Metoda: fine-tuning istniejącej polityki (nie trening od zera) — PPO jest on-policy,
funkcja wartości szybko adaptuje się do zmienionych kar.

**Wyniki fine-tuningu (6M → 8M kroków, reward v2.1)** — porównanie 100 epizodów
na model, identyczna sekwencja randomizacji (ten sam seed → parowane warunki);
`logs/board/eval100_compare.log`:

| Metryka | v2 @ 6M | v2.1 @ 8M | zmiana |
|---|---|---|---|
| Sukcesy (5/5 przycisków) | 87 % | **100 %** | +13 pp |
| Średni czas sukcesu | 522 kroków (26 s) | **352 kroków (18 s)** | −33 % |
| Epizody z kontaktem z płytą | 95 % | 61 % | −34 pp |
| Średnio kroków kolizji / epizod | 15.8 | **2.4** | −85 % |
| Gładkość (śr. ‖Δa‖ / krok) | 3.22 | 3.15 | −2 % |

Wnioski: podniesienie kary kolizji do znaczącego poziomu wyeliminowało wyuczone
"szorowanie" po panelu (pozostały kontakt ~2 kroki/epizod to głównie moment samego
wciskania przycisku przy płycie), a usunięcie zbędnych kontaktów poprawiło też
niezawodność (100/100) i szybkość (−33 %). Kara za zmianę akcji miała marginalny
wpływ na surową metrykę ‖Δa‖ — szarpanie akcji na poziomie polityki jest w dużej
mierze filtrowane przez kontroler OSC; przy wdrożeniu sprzętowym i tak zalecane
jest dodatkowe ograniczenie prędkości/przyspieszeń po stronie kontrolera Kinovy.
Ostatnie 6 ewaluacji treningu: komplet 100 % sukcesów. Wideo: `v5_finetuned_8M.mp4`
(5/5, epizody 292–456 kroków).

Porównanie v2 vs v3 przy 100k kroków (4.58 vs 2.44) pokazuje **cenę realizmu
obserwacji**: szum percepcji i brak podglądu wciśnięć spowalniają zbieżność — to
oczekiwany koszt polityki zdolnej do transferu.

## 7. Napotkane problemy techniczne i rozwiązania

Ta sekcja dokumentuje problemy implementacyjne — materiał do rozdziału o implementacji.

### 7.1. Robosuite `edit_model_xml` przepisuje ścieżki assetów (krytyczne)
Robosuite szuka segmentu "robosuite" w ścieżkach mesh/tekstur i przepisuje je na
katalog zainstalowanego pakietu. Katalog projektu nazywa się `robosuite`, więc ścieżki
własnych assetów były niszczone. **Rozwiązanie:** override `edit_model_xml` w każdym
env — przepisywane są tylko ścieżki zaczynające się od katalogu pakietu.

### 7.2. Jednostki meshy STL (pułapka)
`linear_board.stl`, `50mm_pipe.stl` — w milimetrach (scale 0.001); **`Omniboard.stl`
i `Emergency stop.stl` — w metrach (scale 1)**. Załadowanie grzybka ze scale 0.001
dawało obiekt o rozmiarze 0.07 mm — "niewidzialny" bez komunikatu błędu. Wniosek:
zawsze weryfikować bbox mesha (parsing STL: 80 B nagłówka + rekordy 50 B:
12×float32 + 2 B atrybutu).

### 7.3. Mesh E-Stopa jako jedna bryła
`Emergency stop.stl` zawiera obudowę i grzybek w jednej bryle → nie da się nim pokazać
ruchu części ruchomej. **Rozwiązanie:** przycisk zbudowany z prymitywów (box + walce),
wizualnie 1:1 ze zdjęciem z RoboCup.

### 7.4. Samokolizja grzybek–obudowa
Walce kolizyjne grzybka i obudowy przenikały się konstrukcyjnie → kontakt wypychał
grzybek 5.3 mm poza zakres (przy progu aktywacji 12 mm!). **Rozwiązanie:** maski
`contype/conaffinity` (2/1 i 4/1) — patrz 4.3. Po poprawce spoczynek = 0.0 mm.

### 7.5. `duplicate_collision_geoms` zasłania wizuale
Robosuite domyślnie duplikuje geomy kolizyjne jako widoczne szare bryły — zasłaniały
mesh boardu i przyciski. **Rozwiązanie:** `duplicate_collision_geoms=False` dla
obiektów mających własne geomy wizualne.

### 7.6. Segfault w `PPO.load` — konflikt TensorFlow ↔ `torch._dynamo`
Import `stable_baselines3` ciągnie przez tensorboard cały TensorFlow (obecny w conda
env); załadowany TF psuje późniejszy lazy-import `torch._dynamo` wykonywany przy
tworzeniu optymalizatora Adam wewnątrz `PPO.load` → **segmentation fault**. Świeże
`PPO()` w treningu przeżywało przez inną kolejność importów. Diagnoza:
`python -X faulthandler` + testy izolacyjne (torch.load sam ✓, sb3+dynamo ✗).
**Rozwiązanie:** `sys.modules["tensorflow"] = None` przed importem SB3 (tensorboard
używa własnego stuba) — we wszystkich skryptach używających SB3.

### 7.7. Randomizacja per epizod w `_load_model`
Robosuite z `hard_reset=True` (domyślne) wywołuje `_load_model` przy każdym resecie —
dzięki temu losowanie pozycji boardu w `_load_model` działa per epizod. Przy
`hard_reset=False` ta randomizacja by zamarła (pułapka na przyszłość).

### 7.8. Kamery w arenie
Kamerę trzeba dodać do worldbody areny **przed** utworzeniem `ManipulationTask`
(merge XML); po kompilacji pozę kamery można zmieniać przez
`sim.model.cam_pos/cam_quat` (kwaternion wxyz).

## 8. Narzędzia pomocnicze

- `training/watch_board.py` — podgląd polityki: okno na żywo albo `--record plik.mp4`
  (render offscreen EGL + ffmpeg, działa bez ekranu/przez SSH); auto-wybór modelu
  (best_model → najnowszy checkpoint), `--random`, `--episodes N`.
- `tensorboard --logdir logs/tensorboard` — krzywe uczenia.
- Wznawianie treningu: `python training/train_board.py --resume <checkpoint> --timesteps N`
  (uwaga: N = liczba **dodatkowych** kroków; SB3 zapisuje w checkpointach też stan
  optymalizatora, wznowienie jest czyste w obrębie tej samej wersji env/obs).

## 9. Dalsze kroki

1. Reward v2 (rozdz. 4.6) + trening od zera (~5.5 h).
2. Ocena: success rate w eval (nie sam reward), wideo z `--record`.
3. Kalibracja szumu percepcji do zmierzonego błędu rzeczywistego systemu (AprilTag/YOLO).
4. Pipeline sprzętowy: estymacja pozy boardu → mapowanie obs → polityka → OSC Kinovy;
   detekcja wciśnięcia z siły+kinematyki; maszyna stanów nadzorcy.
5. (Opcjonalnie) porównanie AprilTag vs YOLO jako rozdział pracy.

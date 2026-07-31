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

### 5.1. Architektura nadzorcy (supervisor)

Zasada projektowa: **nadzorca na sprzęcie to dokładnie to, co w symulacji robi
`ButtonBoardEnv` poza fizyką**. Polityka jest tylko funkcją obs→akcja; maszynę
stanów celu, detekcję wciśnięć i budowanie obserwacji env „udaje" wewnętrznie —
na robocie realizuje je deterministyczny program. RL odpowiada wyłącznie za ruch,
nigdy za logikę zadania i bezpieczeństwo (nadzorca = audytowalna skorupa
bezpieczeństwa wokół wyuczonej polityki).

Jedna pętla o częstotliwości sterowania z symulacji (20 Hz, `control_freq`):

```
┌─ raz na start epizodu ──────────────────────────────────────┐
│ kamera → AprilTag → poza 6D boardu                          │
│ → pozycje spoczynkowe i osie 5 przycisków (tabela           │
│   BUTTON_FACES z env) → transformacja do bazy robota        │
└─────────────────────────────────────────────────────────────┘
┌─ co cykl (20 Hz) ───────────────────────────────────────────┐
│ 1. odczyt z Kinovy: stawy, poza EEF, estymata sily          │
│ 2. detektor wcisniecia aktualnego celu                      │
│ 3. maszyna stanow: wcisnieto? → nastepny cel / DONE         │
│ 4. budowa obs: identyczny wektor jak w sim                  │
│ 5. polityka.predict(obs) → 7-dim akcja                      │
│ 6. warstwa bezpieczenstwa → komenda kartezjanska (Kortex)   │
└─────────────────────────────────────────────────────────────┘
```

Klocki i ich odpowiedniki w środowisku symulacyjnym:

| Klocek na sprzęcie | Odpowiednik w `ButtonBoardEnv` |
|---|---|
| estymacja pozy boardu (AprilTag, raz/epizod) + geometria CAD | `_load_model` + `BUTTON_FACES`; błąd estymacji = `percep_pos_bias`, jitter re-estymacji = `percep_pos_noise` |
| detektor wciśnięcia (siła + kinematyka w TCP, debounce) | `ACTIVATION_DEPTH` + bramka `_finger_on_cap` (v2.2) |
| maszyna stanów: lista celów, indeks, DONE/TIMEOUT/FAULT | `_order`, `_current_idx`, `_pressed`, `_check_success` |
| budowa wektora obs (bit w bit jak w sim) | `_setup_observables` + `flatten_obs` |
| warstwa bezpieczeństwa (poza polityką) | brak — w sim zbędna, na sprzęcie obowiązkowa |

**Detektor wciśnięcia** — dwa warunki jednocześnie, utrzymane przez 3–5 kolejnych
cykli (debounce):
- *głębokość z kinematyki*: rzut (poza_EEF − pozycja_spoczynkowa_grzybka) na oś
  wciskania ≥ ~10–12 mm; pozycja spoczynkowa z pozy boardu + CAD, TCP skalibrowany
  na czubek **zamkniętych** palców (stąd wymuszenie zamkniętego grippera w v2.2),
- *siła wzdłuż osi*: Kortex API udostępnia estymatę zewnętrznego wrencha na
  narzędziu (`tool_external_wrench`, liczoną z momentów stawów — nie trzeba
  własnego J^T·τ); sprężyna 500 N/m × 12 mm ≈ 6 N → próg ~4–5 N.

To fizyczny odpowiednik bramki v2.2: wciśnięcie nadgarstkiem nie przejdzie przez
detektor (głębokość liczona w TCP się nie zgadza), więc symulacja musi odrzucać
dokładnie te same wciśnięcia — inaczej polityka trenuje na zachowaniach, których
nadzorca nigdy nie zaliczy.

**Budowa obserwacji** — najbardziej podatny na błędy element: wektor musi być
zgodny z symulacją co do porządku pól, układu współrzędnych (świat sim = baza
robota w (0, 0, 0.65), +X ku ścianie) i konwencji kwaternionów. Na sprzęcie NIE
dodaje się sztucznego szumu — realna percepcja szumi sama; szum w treningu był
po to, by polityka była na to odporna. Zadanie kalibracyjne przed wdrożeniem:
zmierzyć realny błąd AprilTag (np. dotknięcie przycisku ręcznie prowadzonym
robotem vs predykcja) i dopasować `percep_pos_bias`/`percep_pos_noise` w sim
do zmierzonych wartości (ew. krótki dotrening).

**Warstwa bezpieczeństwa** (poza polityką, nadrzędna): box przestrzeni roboczej
wokół boardu, limity prędkości/przyspieszeń po stronie kontrolera (filtr OSC
z symulacji nie przenosi się na sprzęt), próg siły → natychmiastowy stop,
fizyczny E-Stop operatora. Stany awaryjne maszyny: TIMEOUT (brak postępu →
powrót do pozycji domowej), FAULT (przekroczona siła → stop).

Zakres implementacji: jeden proces (Python + Kortex API lub node ROS2, zależnie
od stacku labu), szacunkowo 300–500 linii, z czego większość to detektor
i budowa obs. Testowalny bez robota: zamiast Kortexa podpina się symulację
(ten sam interfejs odczyt/komenda), co pozwala zweryfikować całą pętlę nadzorcy
przed pierwszym uruchomieniem na sprzęcie.

### 5.2. Implementacja nadzorcy i test akceptacyjny na SimIO (2026-07-19)

Nadzorca ZAIMPLEMENTOWANY jako moduł `supervisor/` (~550 linii):
`interfaces.py` (RobotIO — jedyna granica nadzorca↔sprzęt: odczyt stanu TCP/
siły + komenda kartezjańska), `sim_io.py` (ButtonBoardEnv w miejscu Kortexa,
siła zewnętrzna z kontaktów MuJoCo na palcach), `perception.py` (poza boardu
raz na epizod + CAD → pozycje/osie celów, szum jak w treningu),
`press_detector.py`, `state_machine.py` (sekwencja + RetryManager),
`obs_builder.py` (wektor obs w kolejności kluczy z treningu; część „object"
liczy nadzorca, proprio daje robot), `safety.py`, `loop.py` (pętla 20 Hz).
Test: `training/eval_supervisor.py` — polityka dostaje WYŁĄCZNIE obs
zbudowane przez nadzorcę; env służy za fizykę i za prawdę do oceny.

**Wyniki (100 sparowanych epizodów, v2.3 best @ 10M):**

| Konfiguracja | Sukcesy | Śr. czas sukcesu |
|---|---|---|
| Polityka bezpośrednio (referencja, rozdz. 6.3) | 85/100 | 22.9 s |
| Nadzorca, bez retry, horyzont 50 s | 84/100 | 24.4 s |
| Nadzorca + retry (timeout 20 s), horyzont 50 s | 86/100 | 25.0 s |
| **Nadzorca + retry, horyzont 75 s** | **92/100** | 27.3 s |

(final @ 12M analogicznie: 87 → 81 → **92/100** przy 75 s). Wnioski:
(1) **test akceptacyjny zaliczony** — pętla nadzorcy odtwarza wynik polityki
(84≈85) przy pełnej synchronizacji detektor↔fizyka (458=458 wciśnięć);
obs builder jest więc gotowy pod sprzęt. (2) **Retry działa i nic nie psuje**
(2 epizody uratowane, 0 straconych przy 50 s), a WSZYSTKIE pozostałe porażki
wykorzystywały pełny horyzont z aktywnym retry — są duszone czasem, nie
nieodwracalne; przy budżecie 75 s (na sprzęcie limitu nie ma) sukces rośnie
do **92 %** dla obu modeli. Retry: brak wciśnięcia przez 20 s → odjazd 12 cm
przed cel (P-regulator) → oddanie sterowania polityce.

**Odkrycia z fazy debugowania (istotne pod KortexIO):**
- **Siły wewnętrzne chwytaka**: zamknięte palce Robotiq dociskają się
  wzajemnie ~130 N — naiwna suma sił na palcach wyzwalała force-stop
  i zamrażała robota. Liczyć wyłącznie kontakty palec×otoczenie (dokładnie
  tak działa `tool_external_wrench` Kinovy).
- **Detekcja wciśnięcia z siły+kinematyki TCP jest NIEWYKONALNA dla stylu
  wciskania polityki v2.3** (zbadane na 8 epizodach śladów,
  `logs/board/detector_traces.npz`, 3 warianty reguł): polityka napiera na
  przycisk >20 N przez dziesiątki–setki kroków przed aktywacją (siła nie
  lokalizuje zdarzenia w czasie), dociska różnymi fragmentami palców z TCP
  do 10 cm od osi (głębokość absolutna: błąd ±37 mm), a skok grzybka nie
  przenosi się na ruch TCP (delta głębokości od kontaktu: pominięte
  23–33/36 wciśnięć). KONSEKWENCJA: na sprzęcie detekcją pierwszego wyboru
  staje się **zatrzask prawdziwego E-Stopa** (spadek siły po zatrzaśnięciu /
  weryfikacja wizyjna ~10 mm niżej — dawny „plan B"); w testach SimIO
  reprezentuje go `SimTruthDetector`. Docelowa poprawka treningowa:
  reward v2.4 z bramką centrowania TCP na osi przycisku („wykrywalny
  z konstrukcji" styl docisku).
- **Force-stop**: próg 25 N poprawny dla sprzętu (sprężyna ~6 N) masowo
  wyzwalał się w sim przy legalnych dociskach — siły kontaktowe MuJoCo
  (80–440 N) są nierealistyczne; progi siłowe trzeba stroić na sprzęcie,
  nie w symulacji.

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
na model (`logs/board/eval100_compare.log`). Metodologia: obie serie startowały
z tego samego seeda globalnego, ale strumień losowości rozjeżdża się od 2. epizodu
(szum percepcji losowany per krok, a modele robią różną liczbę kroków) — czyli są
to dwie niezależne próby z tej samej dystrybucji warunków, nie parowanie
epizod-w-epizod. Przy n=100 i różnicy 87 vs 100 sukcesów wynik jest jednoznaczny
(test Fishera p ≈ 2·10⁻⁴). Do ściśle parowanych porównań (re-seed przed każdym
epizodem → epizod k ma identyczne warunki i sekwencję szumu dla każdego modelu)
służy `training/compare_models.py` (metryki, CSV, test Fishera).

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

**Reward v2.2 — bramka kontaktu palcowego (wdrożona 2026-07-18, trening w toku):**
obserwacja z wideo: model czasami wciska przyciski nie gripperem, lecz częścią
nadgarstkową od boku. Analiza kodu potwierdziła **drugi przypadek reward hackingu**:
bonus za wciśnięcie zależał wyłącznie od pozycji jointa przycisku (obojętne, co
go wgniata), a kara kolizji celowo omijała geomy przycisków — kontakt dowolnej
części robota z grzybkiem był darmowy. Kwantyfikacja (8 epizodów best_model @ 8M
w nowym env): kontakt nie-palcowy z przyciskami w **8/8 epizodów, śr. 138 kroków**
— nawyk znacznie powszechniejszy, niż sugerowało wideo.

Na sprzęcie to dyskwalifikujące z trzech powodów: (1) Kinova Gen3 ma moduł
wizyjny na nadgarstku — dokładnie w miejscu, którym polityka pcha; (2) detekcja
wciśnięcia zakłada kontakt w TCP — głębokość z kinematyki i siła z momentów
stawów (J^T·τ) liczone dla EEF są błędne przy kontakcie nadgarstkiem (inne ramię
dźwigni); (3) nadzorca przełączałby cele na podstawie fikcyjnych detekcji.

Zmiany (`envs/button_board.py`):
- **bramka na wciśnięcie**: aktywacja liczy się tylko przy jednoczesnym kontakcie
  geomu palca grippera (8 geomów z `important_geoms` Robotiq85) z grzybkiem celu —
  odwzorowuje detekcję nadzorcy na sprzęcie (siła/kinematyka w TCP nie zarejestruje
  wciśnięcia nadgarstkiem); bez kontaktu palcowego sekwencja NIE postępuje,
- **kara misuse**: kontakt nie-palcowej części robota z dowolnym przyciskiem
  karany jak kolizja z płytą (COLLISION_COST),
- **wymuszony zamknięty gripper**: `step()` nadpisuje 7. wymiar akcji na „zamknij"
  (wymiar akcji zostaje — checkpoint 8M pasuje do fine-tuningu); sztywny,
  dobrze zdefiniowany punkt kontaktu w palcach, zgodny z TCP.
Nowa metryka w `compare_models.py`: kroki misuse/epizod. Weryfikacja: test
negatywny (wgniecenie 20 mm bez kontaktu palców → brak zaliczenia), test
wymuszenia grippera, oraz 8 epizodów starego modelu w nowym env — **4/8 sukcesów**
(spadek ze 100 %: stary model częściowo polegał na wciskaniu nadgarstkiem;
wciśnięcia palcami są zaliczane poprawnie). Plan: fine-tuning od checkpointu 8M
(+2M kroków), ewaluacja 100 epizodów z metryką misuse.

**Wynik fine-tuningu v2.2 (2026-07-19): porażka — polityka nauczyła się unikać
przycisków.** Trening 8.03M → 10.04M kroków; ewaluacja deterministyczna
poprawiała się z −4.14 (stary model pod karą misuse) tylko do plateau ~−0.8,
epizody stale 1000 kroków. Porównanie 100 sparowanych epizodów (env v2.2):

| Metryka (100 epizodów, env v2.2) | v2.1 @ 8M | v2.2 best @ ~9.7M | v2.2 final @ 10M |
|---|---|---|---|
| Sukcesy 5/5 | **59/100** | 0/100 | 0/100 |
| Epizody z misuse / śr. kroki | 98/100 / 128.8 | 58/100 / 2.8 | 61/100 / 2.8 |
| Śr. kroki kolizji z płytą | 21.6 | 1.6 | 0.8 |
| Porażki z 0/5 wciśniętych | 7 | 60 | 60 |

Model oduczył się karanego kontaktu (misuse −98 %), ale razem z nim oduczył się
**wciskania w ogóle** (60 % epizodów kończy z 0/5). Diagnoza (5 epizodów starego
modelu z logowaniem par geomów): kara misuse odpalała się **podczas poprawnych
docisków palcami** — czubki zamkniętego Robotiq85 mają tuż za sobą wewnętrzne
knuckle i obudowę, które przy 12 mm skoku nieuchronnie muskają grzybek/bazę
(184 kroki misuse jednocześnie z palcem na grzybku celu; dominujące pary:
`inner_knuckle × cap/base`). Gradient „nie dotykaj przycisków wcale"
(−5/krok, gęsty) był silniejszy niż rzadki bonus +25 — wróciło optimum unikania
znane z rewardu v1, tym razem wywołane **nadmiarową karą**, nie brakiem nagrody.

**Reward v2.3 (wdrożony 2026-07-19): kara misuse zawężona do ogniw ramienia**
(`robot0_*`); kontakt chwytaka (knuckle, obudowa) z przyciskami jest wolny od
kary — towarzyszy poprawnym dociskom i jest bezpieczny na sprzęcie (to bramka
palcowa, nie kara, odwzorowuje detekcję TCP i ona zostaje bez zmian).
Weryfikacja na starym modelu (5 epizodów): kroki misuse 995 → 77, wyłącznie
pary `robot0_b_collision × btn0` (prawdziwe wciskanie nadgarstkiem przycisku
frontowego); nakładka misuse↔poprawny docisk 184 → 3 kroki. Fine-tuning
wznowiony od checkpointu v2.1 @ 8M (+2M kroków). Artefakty nieudanego
przebiegu: `logs/board_v22_failed_misuse/`.

**Wynik fine-tuningu v2.3 (2026-07-19, 8M → 10M):** ewaluacje treningowe rosły
przez cały przebieg (−1.44 na starcie → ~+0.3–0.6 w końcówce, max 0.611) —
w przeciwieństwie do płaskiego plateau v2.2. Porównanie 100 sparowanych
epizodów (env v2.3, `eval100_v23.log/csv`):

| Metryka (100 epizodów, env v2.3) | v2.1 @ 8M | **v2.3 best (~10M)** | v2.3 final @ 10M |
|---|---|---|---|
| Sukcesy 5/5 | 59/100 | **85/100** | 78/100 |
| Średni czas sukcesu | 29.5 s | **22.9 s** | 25.1 s |
| Śr. kroki kolizji z płytą | 21.6 | **3.2** | 3.5 |
| Śr. kroki misuse (ramię×przycisk) | 17.4 | **4.3** | 3.5 |
| Porażki z 0/5 wciśniętych | 7 | 3 | 2 |

Fisher exact (sukcesy): stary vs best p ≈ 6.7·10⁻⁵, stary vs final p ≈ 5.9·10⁻³.

**Kontrola: trening od zera w env v2.3 (2026-07-20, 3M kroków):** 0/20
sukcesów, 0/5 przycisków w każdym epizodzie; najlepszy checkpoint ewaluacyjny
to polityka „nie rób nic" (śr. ‖Δa‖ = 0.02 — stanie w miejscu minimalizuje
kary przy braku odkrytego bonusu). Ilustruje to wartość transferu z linii
fine-tune: 2M kroków od checkpointu v2.1 daje 85 %, podczas gdy 3M od zera
nie odkrywa jeszcze wciskania (bramka palcowa zawęża eksplorację; historyczny
run v2 potrzebował ~6M w łatwiejszym env). Model: `ppo_board_scratch_3M.zip`;
uczciwe porównanie „scratch vs fine-tune" wymagałoby ~8–10M kroków.
Misuse starego modelu wg definicji v2.3 to 17.4 kroku/epizod (nie 128.8 — tamta
liczba obejmowała dozwolony teraz kontakt chwytaka); v2.3 redukuje go o 75 %
przy jednoczesnym wzroście sukcesów 59 → 85 i utrzymaniu niskich kolizji.
Model wciska więc przyciski zamkniętym chwytakiem od frontu, a sekwencja
postępuje wyłącznie po wciśnięciach zaliczonych bramką palcową — zachowanie
zgodne z detekcją sprzętową.

**Kontynuacja fine-tuningu do 12M (2026-07-19): plateau ~85–87 %.** Dalsze
+2M kroków nie dało istotnej poprawy (`eval100_v23_12M.log/csv`): best nowego
przebiegu 85/100 (czas 22.8 s), final 87/100 (25.7 s, ale kolizje 6.2 vs 3.2
kroku u best @ 10M) — różnice w granicach szumu. Ewaluacje treningowe
oscylowały na 0.3–0.6 bez trendu. Kluczowa obserwacja z analizy sparowanej
(identyczne warunki epizodów dzięki re-seedowaniu): best @ 10M i final @ 12M
mają tylko **2 wspólne porażki na 100 epizodów** — zbiory błędów niemal
rozłączne (13 vs 11 unikatowych), a hipotetyczna „wyrocznia" z obu polityk
osiągnęłaby 98/100. Wniosek: nie istnieją systematycznie „trudne" konfiguracje
(pozycja boardu × kolejność × szum percepcji); każda polityka ma własne,
losowo rozmieszczone słabe punkty, a porażki to timeouty w trakcie sekwencji
(0 × stuck, 0 × 0/5 u finala — model zawsze robi postęp). Praktyczne
implikacje: (1) dalszy fine-tuning w tej linii ma malejące zwroty — sensowną
alternatywą byłby świeży trening od zera z rewardem v2.3 (~10M kroków, ~20 h);
(2) na sprzęcie nadzorca może tanio podnieść skuteczność mechanizmem retry
(brak postępu przez N s → odjazd i ponowny dojazd do celu), skoro porażki
są przejściowe, a nie terminalne. Modele: `best_model_v23_10M.zip` (85 %,
najmniej kolizji — kandydat na sprzęt), `ppo_board_final.zip` @ 12M (87 %,
zawsze ≥1 przycisk — kandydat do wideo/statystyk).

Porównanie v2 vs v3 przy 100k kroków (4.58 vs 2.44) pokazuje **cenę realizmu
obserwacji**: szum percepcji i brak podglądu wciśnięć spowalniają zbieżność — to
oczekiwany koszt polityki zdolnej do transferu.

### 6.4. Scratch @ 8M porzucony; nowa wada v2.3 — brak gracji kontaktu (2026-07-20)

Kontynuacja treningu od zera do ~8M (uruchomiona ręcznie po wyniku 0/20 @ 3M,
rozdz. 6.3) **bez postępu** — polityka utknęła w tej samej degeneracji
(„nie rób nic", widoczne na `watch_board.py` jako nadgarstek kręcący się
w miejscu). Decyzja: scratch porzucony, zostajemy przy fine-tune v2.3.
Artefakty zarchiwizowane: `logs/board/checkpoints_scratch_3M_to_8M/`,
`ppo_board_scratch_8M.zip`, `best_model_scratch_8M.zip`; wskaźniki
`ppo_board_final.zip` i `best_model/best_model.zip` (nadpisane przez ten
przebieg) przywrócone do modelu `ppo_board_v23_12M.zip`.

Obserwacja użytkownika przy oglądaniu modelu v2.3 (fine-tune): brak precyzji
w klikaniu — gripper obija się o przyciski z różnych stron zamiast podejść
czysto, a zamknięty chwytak w chwili kontaktu wygląda, jakby jeden szczęk był
dociśnięty aż do nadgarstka, zamiast spoczywać w pozycji, w której oba się
stykają. Na sprzęcie to ryzyko rozjustowania/uszkodzenia chwytaka Robotiq
i zarysowań obudowy przycisków przy powtarzalnych uderzeniach.

**Kwantyfikacja (6 epizodów, `ppo_board_v23_12M.zip`):**
- kontakty palca z przyciskiem INNYM niż aktualny cel: 512 kroków vs 1511
  kroków kontaktu z celem — **~25% kontaktów palcowych to obijanie się
  o niewłaściwy przycisk** po drodze do celu
  *[KOREKTA 2026-07-21: ta liczba okazała się artefaktem pomiaru — ~93%
  tych kroków to zalegający kontakt z dopiero-co-wciśniętym przyciskiem
  podczas odjazdu (cel przeskakuje na następny w kroku sukcesu), a nie
  obijanie; prawdziwe obijanie o niewciśnięte przyciski to ~2% kontaktów.
  Ten sam artefakt w definicji kary okazał się przyczyną zapaści
  fine-tune v2.4 — szczegóły rozdz. 6.5-6.6]*,
- prędkość EEF w chwili kontaktu palec–przycisk: do **1.0 m/s** (szybkie
  uderzenie, nie kontrolowane dosunięcie),
- asymetria stawów palców (`finger_joint`/`right_outer_knuckle_joint` vs
  sprzężone stawy wewnętrzne) >0.15 rad podczas kontaktu w **6/6 epizodów** —
  potwierdza obserwację wizualną: przy uderzeniu pod kątem miękki regulator
  pozycji siłowników (kp=20) przegrywa z siłą kontaktu i szczęki chwytaka
  rozjeżdżają się asymetrycznie zamiast trzymać zamkniętą pozycję.

**Diagnoza:** to bezpośredni efekt uboczny naprawy v2.3 — zwolnienie kontaktu
chwytaka z kary (konieczne, by cofnąć kolaps „unikaj wszystkiego" z v2.2)
usunęło WSZELKĄ zachętę do delikatności. Reward nagradza wyłącznie finalny
kontakt palca z grzybkiem celu; nic nie karze szybkiego, chaotycznego dojazdu
ani obijania się o przyciski po drodze.

**Proponowany reward v2.4 (niewdrożony, wymaga kolejnego fine-tuningu
~2M kroków):**
1. kara za kontakt palca z przyciskiem INNYM niż aktualny cel (analogia do
   kary misuse na ramieniu, ale dla chwytaka — celuje bezpośrednio
   w obijanie się o niewłaściwe przyciski),
2. kara za prędkość EEF w chwili kontaktu z jakimkolwiek przyciskiem powyżej
   progu (np. 0.15–0.2 m/s) — wymusza spowolnienie przed dotknięciem,
3. (opcjonalnie, taniej) zwiększenie wagi `eef_align_with_target` — silniejsza
   zachęta do prostopadłego podejścia od początku ruchu.
Asymetria stawów nie wymaga poprawki w kodzie — to oczekiwany skutek fizyki
kontaktu przy miękkim sterowaniu pozycją (kp=20) chwytaka Robotiq, ustąpi
naturalnie wraz ze spadkiem prędkości/częstości uderzeń po v2.4.

### 6.5. Reward v2.4 wdrożony i fine-tune 12M→15M: PORAŻKA (2026-07-21)

Wdrożono dokładnie propozycję z rozdz. 6.4 (`envs/button_board.py`):
`FINGER_MISUSE_COST=2.0` za kontakt palca z przyciskiem innym niż cel,
`CONTACT_SPEED_COST=5.0` za nadwyżkę prędkości EEF powyżej
`CONTACT_SPEED_THRESH=0.15` m/s w chwili JAKIEGOKOLWIEK kontaktu palec–
–przycisk, oraz podniesiono `DENSE_ALIGN_COEF` 0.05→0.15. Fine-tune od
`ppo_board_v23_12M.zip`, 3M kroków (12M→15M), ~4h, bez interwencji w trakcie.
Modele: `ppo_board_v24_3M.zip` (final), `best_model_v24_3M.zip` (best, checkpoint
z najwyższym eval reward w trakcie treningu).

**Porównanie sparowane, 100 epizodów, `compare_models.py`
(`eval100_v24_vs_v23.csv`):**

| model | sukces | śr. czas sukcesu | epizody z kolizją | kolizje/ep | epizody z misuse | misuse/ep |
|---|---|---|---|---|---|---|
| v2.3 best@10M | **85/100** | 487 kr. | 54/100 | 1.9 | 70/100 | 3.2 |
| v2.3 final@12M | **90/100** | 440 kr. | 72/100 | 4.8 | 54/100 | 5.1 |
| v2.4 best@3M | **70/100** | 523 kr. | 64/100 | 2.7 | 54/100 | 3.6 |
| v2.4 final@3M | **43/100** | 646 kr. | 44/100 | 3.2 | 48/100 | 3.4 |

Sukces spadł względem punktu startowego (v2.3 best@10M 85% → v2.4 best@3M
70%) i **pogarszał się dalej z kolejnymi krokami fine-tuningu w tym samym
przebiegu** (best@3M 70% → final@3M 43%) — to trend, nie szum: ten sam wzorzec
co w kolapsie v2.2 (rozdz. 6.3), tylko łagodniejszy (polityka nie przestała
naciskać całkowicie, ale coraz częściej rezygnuje z dokończenia sekwencji —
rozkład porażek wg liczby wciśniętych przycisków przesuwa się w stronę 0-1/5
w miarę treningu).

**Diagnostyka gracji kontaktu, 25 epizodów/model, ten sam seed
(`diag_contact_grace.py`, `diag_grace_v24_vs_v23.log`)** — sprawdza, czy
problem z rozdz. 6.4 (obijanie się o zły przycisk, wysoka prędkość kontaktu,
asymetria szczęk) w ogóle się poprawił:

| model | obijanie się (zły przycisk) | kontakty >0.15 m/s | asymetria szczęk >0.15 rad |
|---|---|---|---|
| v2.3 best@10M | 24.6% | 58.0% | 88.2% |
| v2.3 final@12M | 26.4% | 51.9% | 90.8% |
| v2.4 best@3M | 24.2% | 53.7% | 95.0% |
| v2.4 final@3M | 21.6% | 57.1% | 89.7% |

**Wszystkie trzy metryki nieodróżnialne od v2.3 w granicach szumu 25-epizodowej
próby** — v2.4 NIE poprawił gracji kontaktu, mimo spadku sukcesu o 15-47 pp.
*(Uwaga: kolumna "obijanie się" w tej tabeli jest liczona starą, wadliwą
definicją — obejmuje zalegający kontakt po sukcesie, patrz diagnoza niżej;
do porównań względnych między modelami nadaje się, bo wada jest wspólna,
ale wartości bezwzględne są ~13× zawyżone.)*

**Diagnoza przyczyny — pierwotna (2026-07-21 rano, NIEPEŁNA):**
`CONTACT_SPEED_COST` karał prędkość przy **każdym** kontakcie palec–przycisk,
łącznie z kontaktem z celem, więc karał też poprawne szybkie dociśnięcia.

**Diagnoza właściwa (2026-07-21, zweryfikowana empirycznie — rozdz. 6.6):**
główny mechanizm zapaści był inny i głębszy. Maszyna stanów przeskakuje cel
na następny przycisk **w tym samym kroku**, w którym pada bonus za wciśnięcie
— a palec jeszcze przez kilkanaście kroków dotyka dopiero-co-wciśniętego
grzybka podczas odjazdu. Ten zalegający kontakt liczył się jako "kontakt
z NIE-celem": pomiar na 8 epizodach v2.3 wykazał, że **~93% wszystkich kroków
"złego kontaktu" to właśnie ten artefakt** (156/168 kroków; prawdziwe
obijanie o niewciśnięte przyciski: 12 kroków, ~2% kontaktów). Skutek: kara
misuse + kara prędkości (odjazd jest szybki) odpalały się **natychmiast po
każdym udanym wciśnięciu**, systematycznie "skubiąc" bonus +25 — polityka
uczyła się, że wciskanie się nie opłaca, tym mocniej im dłużej trenowała
(85%→70%→43%). Ten sam mechanizm co kolaps v2.2 (kara sprzężona z zachowaniem
docelowym), tylko wolniejszy. To tłumaczy też, czemu metryki gracji się nie
zmieniły: sygnał kary prawie wcale nie dotyczył prawdziwego obijania.

**Decyzja: v2.4 odrzucony.** Model do sprzętu/nadzorcy: `ppo_board_v23_12M.zip`
(final@12M, 90% sukces) lub `best_model_v23_10M.zip` (85% sukces, mniej
kolizji). Backupy v2.4: `ppo_board_v24_3M.zip`, `best_model_v24_3M.zip`;
wideo: `video_v24_best3M.mp4`, `video_v24_final3M.mp4`; nowe narzędzie:
`training/diag_contact_grace.py`.

*Errata porządkowa: wcześniejsza wersja tego rozdziału twierdziła, że
wskaźniki live nie zostały nadpisane — nieprawda. Nocny trening v2.4 nadpisał
na końcu `ppo_board_final.zip` (zapis końcowy), a krótki wadliwy start v2.5
nadpisał `best_model/best_model.zip` (EvalCallback). Oba wykryte po hashach
md5 i przywrócone do `ppo_board_v23_12M.zip` 2026-07-21.*

### 6.6. Reward v2.5 — naprawa właściwej przyczyny i fine-tune (2026-07-21)

Pierwsza wersja v2.5 (uruchomiona rano 21.07) ograniczała jedynie karę
prędkości do kontaktu z nie-celem — czyli naprawiała diagnozę pierwotną,
ale NIE artefakt zalegającego kontaktu (kontakt z dopiero-co-wciśniętym
przyciskiem nadal liczył się jako "zły"). Trening z tą wersją zatrzymano po
~226k kroków (przebieg odrzucony; checkpointy zarchiwizowane w
`checkpoints_v24_v25buggy/` razem z checkpointami v2.4, kopia wadliwego
best: `best_model_v25buggy_226k.zip`).

**Weryfikacja empiryczna artefaktu** (8 epizodów `ppo_board_v23_12M.zip`,
seed 4242): 168 kroków "złego kontaktu", z czego 156 (92.9%) to kontakt
wyłącznie z już-wciśniętymi przyciskami (odjazd po sukcesie), 12 (7.1%) to
prawdziwe obijanie. Kara prędkości przy zaleganiu: 72 kroki; przy prawdziwym
obijaniu: 3 kroki. Wniosek: w v2.4 **96% kar prędkości przy "złym kontakcie"
było wymierzonych w poprawny odjazd po udanym wciśnięciu**.

**Ostateczna definicja v2.5** (`envs/button_board.py`): obie kary
(`FINGER_MISUSE_COST`, `CONTACT_SPEED_COST`) odpalają się wyłącznie przy
kontakcie palca z przyciskiem, który (a) nie jest bieżącym celem **i**
(b) nie został jeszcze wciśnięty. Zalegający kontakt po sukcesie oraz
poprawny docisk celu (nawet szybki) są wolne od kar — opodatkowanie
zachowania docelowego zawaliło już dwa fine-tune'y (v2.2, v2.4). Kontakt
celu pozostaje świadomie niekarany; ewentualna kara za prędkość dojazdu
do celu (strefa <3 cm przed kontaktem) to pomysł na v2.6, jeśli sam
sygnał anty-obijaniowy nie wystarczy.

Poprawiono też `training/diag_contact_grace.py`: metryka obijania liczy
teraz wyłącznie kontakt z niewciśniętym nie-celem, zaleganie raportowane
osobno. Skorygowany punkt odniesienia dla v2.3 (final@12M, 4 ep.):
obijanie 1.6% kontaktów, zaleganie 15.9%.

**Wyniki fine-tune v2.5 (12M→15M, 3M kroków, 2026-07-21).** Porównanie
sparowane 100 epizodów, seed 12345 — wprost porównywalne z tabelą w 6.5
(te same warunki epizodów):

| model | sukces | śr. czas sukcesu | kolizje kroki/ep | misuse kroki/ep |
|---|---|---|---|---|
| v2.3 final@12M (baza) | **90/100** | 440 kr. | 4.8 | 5.1 |
| v2.3 best@10M | 85/100 | 487 kr. | 1.9 | 3.2 |
| v2.4 best@3M | 70/100 | 523 kr. | 2.7 | 3.6 |
| **v2.5 best@3M** | **83/100** | 510 kr. | 6.8 | 6.7 |
| **v2.5 final@3M** | **61/100** | 597 kr. | 8.2 | 8.3 |

Diagnostyka gracji (poprawiona metryka, 25 epizodów, seed 55555,
`diag_grace_v25.log`): obijanie o niewciśnięty nie-cel v2.3 1.8% → v2.5 best
1.4% / final 3.6% (szum przy tak niskiej bazie); prędkość kontaktu
>0.15 m/s: 51.9% → 45.3% / 55.5%; asymetria szczęk: bez zmian (~90%).

**Wnioski:**
1. Naprawa artefaktu zadziałała *częściowo*: v2.5 best (83%) nie zapadł się
   jak v2.4 best (70%), ale nadal nie przebił bazy (90%), a final znowu
   degraduje względem best (61%, Fisher p=8.5e-4) — łagodniejszy, lecz ten
   sam trend spadkowy z czasem treningu. Kolizje i misuse ramienia wręcz
   wzrosły (4.8→6.8-8.2 kroków/ep).
2. Gracja kontaktu nie poprawiła się mierzalnie — co po korekcie metryki
   przestało dziwić: prawdziwe obijanie to zaledwie ~2% kontaktów, więc kara
   anty-obijaniowa niemal nie ma na czym pracować; a głównych składników
   "braku gracji" (szybki docisk CELU, asymetria szczęk przy uderzeniu) v2.5
   świadomie nie karze, bo opodatkowanie zachowania docelowego dwukrotnie
   zawaliło politykę.
3. **Decyzja: v2.5 odrzucony, KONIEC prób naprawy gracji przez reward
   (v2.2, v2.4, v2.5 — trzy porażki).** Wskaźniki live przywrócone do
   `ppo_board_v23_12M.zip` (zweryfikowane md5). Backupy: `ppo_board_v25_3M.zip`,
   `best_model_v25_3M.zip`; wideo `video_v25_best3M.mp4`/`video_v25_final3M.mp4`;
   `eval100_v25.csv`, `diag_grace_v25.log`.
4. **Rekomendowana droga dla gracji kontaktu: warstwa nadzorcy, nie reward.**
   `SafetyLayer` w module `supervisor/` już clipuje akcje — naturalne
   rozszerzenie to limit prędkości TCP w strefie przy tablicy (np. <5 cm od
   płaszczyzny przycisków), działający identycznie w symulacji i na sprzęcie,
   bez ryzyka psucia wytrenowanej polityki. To rozwiązanie sim2real-właściwe:
   chroni robota także wtedy, gdy polityka zachowa się nieprzewidzianie.

### 6.7. Robot uczelniany to Gen3 **6DOF** — integracja `Kinova3_6DOF` (2026-07-29)

Ustalenie: fizyczny robot na uczelni to Kinova Gen3 w wariancie **6DOF**,
podczas gdy stockowy `Kinova3` w robosuite (używany we wszystkich dotychczasowych
treningach) to wariant **7DOF**. Różnice istotne dla sim2real: inna kinematyka
(brak redundancji łokcia, zasięg ~0.89 m vs ~0.90 m), o jeden staw mniej
w obserwacjach proprio.

**Integracja (bez modyfikacji site-packages, wszystko w repo):**

- Model MJCF + meshe: `my_assets/robots/kinova3_6dof/` — zaadaptowane z
  [goldengeko/robosuite_training_examples](https://github.com/goldengeko/robosuite_training_examples)
  (inercje z oficjalnego opisu Kortex; momenty 39 Nm duże stawy / 9 Nm nadgarstek).
- Klasa robota: `envs/kinova3_6dof_robot.py`, rejestrowana w **runtime** przez
  metaklasę robosuite + `@register_robot_class("FixedBaseRobot")`; import
  w `envs/__init__.py`, więc działa też w workerach SubprocVecEnv.
- Wybór robota w treningu: klucz `robot` w configu YAML (domyślnie `Kinova3`).

**Poprawki względem repo źródłowego (obie zweryfikowane empirycznie):**

1. **Błąd `init_qpos` w repo:** autor skopiował pozę startową 7DOF obcinając
   ostatni element — ale w 6DOF stawy pochyłu to 2/3/5, a w 7DOF 2/4/6, więc
   zgięcie łokcia (1.89 rad) trafiało w staw obrotowy i ramię startowało
   wyprostowane w górę (EEF z=2.67 m, poza zasięgiem tablicy). Po remapie
   `[0, 0.65, -1.89, 0, -0.6, -π/2]` EEF startuje w [0.427, 0.00, 1.60] —
   niemal identycznie jak 7DOF ([0.433, -0.03, 1.61]).
2. Tłumienie stawów 0 → 0.1 (duże) / 0.01 (nadgarstek), jak w stockowym XML.
3. Parytet sceny: baza `RethinkMount` (nie `SpotFloating` z repo), gripper
   `Robotiq85Gripper` (nie Robotiq140) — jak dotychczas.

**Weryfikacja (smoke testy):** wszystkie przyciski w zasięgu (0.60–0.78 m od
bazy), serwo OSC dojeżdża porównywalnie do 7DOF (odchyłki tego samego rzędu
na tych samych przyciskach), fizyka stabilna (brak NaN, |qvel| ramienia
w normie), render sceny poprawny.

**Konsekwencje — WAŻNE:**

- Wymiar obserwacji spada ze **128 do 120**: joint_pos, joint_pos_cos,
  joint_pos_sin i joint_vel mają po 6 elementów zamiast 7 (4×1 = 8 mniej).
  **Wytrenowane modele (w tym `ppo_board_v23_12M`) są NIEkompatybilne
  z 6DOF** — wejście sieci ma inny wymiar, fine-tune wprost niemożliwy.
- Przestrzeń akcji bez zmian (OSC_POSE 6D delta + gripper = 7), reward i env
  robot-agnostyczne — zmiana sprowadza się do `robot: Kinova3_6DOF` w configu.
- Do decyzji: pełny retrening dla 6DOF (scratch z bramką palcową v2.3 wcześniej
  się nie udał — realistycznie kurikulum jak dla 7DOF: v2.1 → fine-tune v2.3,
  ~10M kroków, ~20 h) albo pozostanie przy 7DOF w symulacji ze świadomością
  różnicy kinematyki przy przejściu na sprzęt (nadzorca wysyła komendy
  kartezjańskie OSC, więc różnica DOF jest częściowo maskowana przez
  kontroler Kortex, ale zachowanie przy ograniczeniach stawów będzie inne).

### 6.8. Retrening 6DOF, etap 1 — pierwsza próba NIEUDANA i diagnoza (2026-07-30)

**Plan kurikulum** (decyzja: pełny retrening dla 6DOF): etap 1 = scratch bez
bramki palcowej (scratch z bramką padał do polityki „nie rób nic", rozdz. 6.4),
etap 2 = fine-tune z bramką (pełny reward v2.3). Configi:
`training/configs/ppo_board_6dof_s1.yaml` / `_s2.yaml`, artefakty w
`logs/board_6dof/` (odseparowane od 7DOF).

**Pierwsza próba etapu 1 (2026-07-29→30, 6M kroków, ~13 h): PORAŻKA.**
Przez całe 6M ani jednego sukcesu ewaluacji (długość zawsze 1000 = pełny
horyzont), reward ewaluacji −0.2…−2.7 bez trendu, „best model" z 700k kroków,
std polityki rosło do 1.6 (polityka coraz bardziej losowa — entropia rośnie,
bo nie ma czego eksploatować). Diagnostyka rolloutów: polityka **nigdy nie
zbliżała się poniżej 18–39 cm od celu**, 0 wciśnięć, 0 kolizji, akcje
niemal maksymalne (ep_rew ≈ czysty koszt akcji).

**Wykluczone przyczyny (testy A/B 6DOF vs 7DOF, identyczny kod):**

- Obserwacje: struktura i wartości poprawne (120 wymiarów, `target_relative_pos`
  zgodny z prawdą sim co do 1e-4; kolejność kluczy identyczna jak w 7DOF).
- Sterowanie OSC: skryptowany serwo dojeżdża 0.43 m → 0.044 m w 60 kroków;
  śledzenie pozycji przy jednoczesnych losowych akcjach rotacji **nie gorsze
  niż na 7DOF** (mediana błędu 0.035–0.041 m vs 0.035–0.064 m).
- Fizyczna wykonalność docisku: skryptowany docisk bez sterowania orientacją
  zatrzymuje się na kostkach palców 4.4 cm od grzybka **identycznie na obu
  robotach** — to ograniczenie prymitywnego testu, nie defekt 6DOF
  (wytrenowana polityka 7DOF dociska po nauczeniu orientacji nadgarstka).

**Właściwa przyczyna: reward nie był tym, w którym 7DOF uczył się od zera.**
Udany scratch 7DOF („v4", rozdz. 6.3) biegł z rewardem v2: `COLLISION_COST
0.5` (pomijalna) i bez `ACTION_RATE_COST`; kary 5.0 i 0.02 dołożył dopiero
fine-tune v2.1 na wytrenowanej polityce. Od v2.1 te wartości są jednak
wpisane w env na stałe — więc scratch 6DOF dostał je od pierwszego kroku.
Bilans dla eksploracji: pełny dolot do celu wart ~10·0.7/225 ≈ +0.031,
a **jeden krok kontaktu z tablicą kosztuje 5/225 ≈ −0.022** — losowe
macanie tablicy to dziesiątki takich kroków, więc funkcja wartości uczy się
„nie podchodź do tablicy" zanim eksploracja odkryje bonus +25 za wciśnięcie.
Stąd polityka wisząca w bezpiecznej odległości, bez kolizji i bez wciśnięć.
To samo zjawisko co przy scratchu z bramką palcową (rozdz. 6.4): **kara
(lub brak nagrody) sprzężona z zachowaniem wymaganym do odkrycia zadania
dusi eksplorację** — tym razem w wersji „kara za przedpole celu".

**Naprawa:** `collision_cost` i `action_rate_cost` sparametryzowane w
konstruktorze `ButtonBoardEnv` (domyślne 5.0/0.02 = v2.3 bez zmian;
regresja potwierdzona bit-w-bit na sparowanych epizodach v23_12M) i wpięte
w YAML. Etap 1 powtórzony od zera w warunkach v2: `collision_cost 0.5`,
`action_rate_cost 0.0`, bramka OFF. Kary wracają w etapie 2 razem z bramką
(jak w historii 7DOF: v2 → v2.1 → v2.3). Artefakty nieudanej próby:
`logs/board_6dof/s1_failed_v21_penalties/` (modele, log stdout, wideo
`video_s1_failed_final.mp4` — widać wiszenie z dala od tablicy).

### 6.9. Druga porażka etapu 1 i właściwa przyczyna: błąd modelu kolizji 6DOF (2026-07-30)

Powtórzony etap 1 (reward v2-like) nauczył się dolotu (2–3 cm od celu ~2.5M)
i wgniatania do ~5 mm, ale do 5.2M kroków nie zaliczył ani jednego wciśnięcia —
także po dołożeniu od 3.9M **shapingu głębokości** (`depress_shaping_k`,
potential-based `k·(depr_t − depr_{t−1})`, push płaci / hold 0 / release oddaje;
człon domyślnie wyłączony, regresja v2.3 bit-w-bit czysta).

**Diagnoza (sekwencja wykluczeń):**

1. IK bez kolizji sugerował, że pozy dociskowe są osiągalne — ale teleport do
   tych póz pokazał **przenikania** (IK ignorował kolizje).
2. Skan pozycji bazy (`base_x_offset`, nowy parametr env — sim2real: gdzie
   parkuje Spot) z IK świadomym kolizji: przysuwanie bazy POGARSZA osiągalność
   (ramię musi się ciaśniej składać). Obecne ustawienie (baza 0.50 m od ściany)
   jest optymalne — zostaje.
3. **WŁAŚCIWY BŁĄD:** w XML robota 6DOF z repo źródłowego tylko `base_collision`
   miał `conaffinity="0"`; pozostałe ogniwa miały domyślne `conaffinity=1`,
   podczas gdy konwencja robosuite (stockowy Kinova3) to `conaffinity="0"` na
   WSZYSTKICH collision geomach ramienia (wyłączenie samokolizji ogniw).
   Skutek: **permanentny fantomowy kontakt base×shoulder z penetracją
   −14.4 mm w każdej konfiguracji** — stała siła kontaktowa na barku przez
   cały dotychczasowy trening 6DOF. Po naprawie (conaffinity=0 na 6 geomach)
   błąd wyrównania osi narzędzia w teście serwo z orientacją spadł
   z ~13–14.5° do **2.1–2.4°** — fantomowa siła paraliżowała precyzję
   orientacji nadgarstka.
4. Analiza mechanizmu docisku (kontakty + render): w pełni osiowy docisk
   „szczypcami na wprost" jest **zablokowany geometrią chwytaka na obu
   robotach** — pręty palców (inner_finger) zamkniętego Robotiq85 są szersze
   niż grzybek i opierają się o pierścień przycisku/płytę skosu, zanim opuszki
   wcisną 12 mm (max ~0.5 mm). Wytrenowany 7DOF wciska **skośnie, narożnikiem
   opuszki** — technika wymagająca precyzyjnej kontroli nadgarstka, której
   6DOF z fantomową siłą nie mógł opanować.

**Weryfikacja transferu:** checkpoint 5.2M w naprawionej fizyce od razu
dojeżdża na 1.6–1.7 cm i wgniata **6.7–7.5 mm** (wcześniej sufit ~4.6 mm) —
4.5 mm od progu aktywacji. Trening wznowiony z 5.2M (+3M) na naprawionym
modelu; granica reżimów fizyki w krzywych: **5.2M kroków**.

### 6.10. Próg aktywacji 12 mm fizycznie nieosiągalny dla 6DOF — kalibracja do 10 mm (2026-07-31)

Po naprawie XML (rozdz. 6.9) polityka systematycznie pogłębiała wciśnięcia:
4.6 mm (2.5M) → 7.5 mm (5.2M) → 10.1 mm (6.4M) → plateau 9.3–10.9 mm (7.6M),
techniką **skośnego docisku opuszką** (kontakt `*_fingertip_collision`, siły
na grzybku 6–22 N — sufit siły OSC to nie problem; 12 mm ≈ 6 N + tarcie).
Postęp zamarł 1–2 mm pod progiem.

**Pomiary wykonalności** (skrypty, przycisk btn1 — jedyny z idealnie
osiągalną pozą osiową):

| technika | max wciśnięcie |
|---|---|
| wyśrodkowane szczypce wzdłuż osi | 0.5 mm (pręty palców siadają na pierścieniu) |
| celowy docisk JEDNĄ opuszką (offset boczny 8–20 mm, oś ±2°) | 6.7 mm |
| nauczona technika skośna (narożnik opuszki w krawędź grzybka) | 10.9 mm |

Przy skośnym kontakcie w krawędź duża część siły idzie radialnie (sztywny
przegub), nie w sprężynę — sufit ~11 mm jest wpisany w technikę, a lepszej
techniki geometria chwytak–grzybek w pozach osiągalnych dla 6DOF nie
dopuszcza. Na 7DOF już przy 12 mm „knykieć muskał grzybek" (rozdz. 6.3/v2.2)
— tam ratowała redundancja nadgarstka.

**Decyzja:** `activation_depth` sparametryzowane w env (domyślne 0.012 =
zgodność 7DOF/v2.3, regresja bit-w-bit czysta); dla 6DOF **0.010 m**
(2/3 skoku 15 mm). Uzasadnienie sim2real: 12 mm było arbitralnym wyborem
projektowym symulacji, realna głębokość aktywacji Omniboardu nie jest znana,
a na sprzęcie wciśnięcie wykrywa **elektrycznie nadzorca** — próg w sim
definiuje zadanie treningowe, nie prawdę sprzętową. Do zmierzenia na
sprzęcie przy wdrożeniu. Weryfikacja: polityka @7.7M w env z progiem 10 mm
od razu zalicza 1–2/5 wciśnięć bez retreningu. Trening wznowiony z 7.7M
(+3M); granice reżimów w krzywych eval: 5.2M (fizyka), 7.7M (próg 10 mm).

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
5. Retrening 6DOF W TOKU (rozdz. 6.8): etap 1 (scratch, reward v2-like)
   powtórzony po nieudanej pierwszej próbie; po nim etap 2 (fine-tune
   z bramką palcową, pełny v2.3) + ewaluacja i wideo.
6. (Opcjonalnie) porównanie AprilTag vs YOLO jako rozdział pracy.

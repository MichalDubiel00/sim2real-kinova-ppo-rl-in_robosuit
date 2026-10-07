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
  > **SPROSTOWANIE (6.33, 2026-09-14):** określenie „realistyczna" nie miało źródła.
  > Karta katalogowa E-stopu: **25 N przy skoku 5 mm**. Skok 15 mm, 500 N/m i próg
  > 12 mm były arbitralnym wyborem projektowym.
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

Obserwacja przy oglądaniu modelu v2.3 (fine-tune): brak precyzji
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

### 6.10. Obniżenie progu aktywacji do 10 mm jako element kurikulum (2026-07-31)

Po naprawie XML (rozdz. 6.9) polityka systematycznie pogłębiała wciśnięcia:
4.6 mm (2.5M) → 7.5 mm (5.2M) → 10.1 mm (6.4M) → plateau 9.3–10.9 mm (7.6M),
techniką **skośnego docisku opuszką** (kontakt `*_fingertip_collision`, siły
na grzybku 6–22 N — sufit siły OSC to nie problem; 12 mm ≈ 6 N + tarcie).
Postęp zamarł 1–2 mm pod progiem, przez ~1.5M kroków bez ani jednego
zaliczonego wciśnięcia.

**Decyzja:** `activation_depth` sparametryzowane w env (domyślne 0.012 =
zgodność 7DOF/v2.3, regresja bit-w-bit czysta); dla 6DOF **0.010 m**
(2/3 skoku 15 mm). Uzasadnienie sim2real: 12 mm było arbitralnym wyborem
projektowym symulacji, realna głębokość aktywacji Omniboardu nie jest znana,
a na sprzęcie wciśnięcie wykrywa **elektrycznie nadzorca** — próg w sim
definiuje zadanie treningowe, nie prawdę sprzętową (do zmierzenia na
sprzęcie przy wdrożeniu). Weryfikacja: polityka @7.7M w env z progiem 10 mm
od razu zalicza 1–2/5 wciśnięć bez retreningu; próg odblokował uczenie
(pierwsze wciśnięcia w treningu ~8.5M). Granice reżimów w krzywych eval:
**5.2M** (naprawa fizyki), **7.7M** (próg 10 mm).

> **KOREKTA (2026-07-31, po pomiarach referencyjnych 7DOF).** Pierwotnie
> opisano tu próg 12 mm jako *fizycznie nieosiągalny* dla 6DOF, na podstawie
> baterii testów skryptowanego docisku (max 0.5 mm szczypcami osiowo, 6.7 mm
> celowaną opuszką). **Ta teza jest fałszywa** — wynikała z błędu metody:
> wszystkie skrypty wymuszały orientację narzędzia **wzdłuż osi przycisku**,
> a wytrenowana polityka 7DOF wciska pod kątem **86.5° do tej osi** (pomiar
> w chwili wciśnięcia btn3: oś narzędzia [0.64, −0.53, −0.55] vs oś przycisku
> [0.71, 0, 0.71]) — czyli zahacza krawędzią opuszki *prostopadle*, a nie
> dociska osiowo. Referencja 7DOF na wszystkich pięciu przyciskach:
> 12.2–16.1 mm, wciśnięcie w 55–182 krokach. Wniosek: 12 mm jest osiągalne,
> a właściwym ograniczeniem 6DOF było **nieopanowanie techniki**, nie
> geometria. Obniżenie progu do 10 mm pozostaje uzasadnione **jako element
> kurikulum** (odblokowało sygnał uczący) i jako parametr projektowy zadania,
> ale nie wolno go uzasadniać niewykonalnością fizyczną. Metodologiczna
> lekcja do pracy: *negatywny wynik testu skryptowanego nie dowodzi
> niewykonalności — dowodzi tylko, że testowana strategia jest zła.*

### 6.11. Wyniki etapu 1 (10.7M kroków): technika opanowana, sekwencja nie (2026-07-31)

Etap 1 zakończony na 10.7M kroków. Ewaluacja 50 epizodów (warunki treningowe,
DR + szum percepcji, próg 10 mm, `logs/board_6dof/eval50_s1_final.log`):

| metryka | wynik |
|---|---|
| pełne sukcesy 5/5 | **0/50** |
| rozkład wciśnięć na epizod | 0×: 20, 1×: 17, 2×: 8, 3×: 3, 4×: 2 |
| wciśnięcia per przycisk (50 ep.) | btn0: 14, btn1: 14, btn2: 10, btn3: **0**, btn4: 12 |
| mediana długości epizodu | 1000 (timeout) |

Polityka **umie wciskać** (76 wciśnięć w 50 epizodach, 2 epizody z 4/5), ale
jest zbyt wolna i zawodna, by domknąć sekwencję 5 przycisków w horyzoncie
1000 kroków. Wideo: `logs/board_6dof/video_s1_final_10.7M.mp4`.

**btn3 (dolny skos) — 0/50: dostęp jest, brakuje techniki.** Test dostępności
(IK celujący **opuszką palca**, bez wymogu orientacji osiowej, z kontrolą
penetracji) wykazał bezkolizyjne konfiguracje z opuszką na czapce dla
**wszystkich pięciu** przycisków (btn3: 0.0 mm od czapki, penetracja 0).
Skan wysokości tablicy (`BOARD_Z` 1.05–1.25 m) nie zmienia obrazu btn3 —
to nie jest problem ustawienia sceny. Ponieważ 7DOF wciska btn3 w 55 krokach
(rozdz. 6.10, korekta), przyczyną jest **niedouczenie**: polityka odkryła
technikę docisku dopiero ~8.5M i ma za sobą ~2 M kroków jej eksploatacji,
podczas gdy 7DOF potrzebował ~4.5M od pierwszych sukcesów do 87 %
(rozdz. 6.3). Decyzja: **kontynuacja etapu 1** (+3M, do ~13.7M) przed
przejściem do etapu 2 — fine-tuning z bramką palcową na polityce 0/50
byłby stratą czasu (bramka tylko zaostrza warunek zaliczenia).

### 6.12. Pomiar postępu 10.7M → 12.2M i horyzont jako element kurikulum (2026-07-31)

Kontynuacja etapu 1 nie dała pełnych sukcesów w ewaluacji treningowej, więc
zmierzono postęp bezpośrednio (20 sparowanych epizodów na model, te same
seedy, `scratchpad/eval_progress.py`):

| pomiar | 10.7M, horyzont 1000 | 12.2M, horyzont 1000 | 12.2M, horyzont 2000 |
|---|---|---|---|
| sukcesy 5/5 | 0/20 | 0/20 | 0/20 |
| wciśnięcia razem | 19 (0.95/ep) | **27 (1.35/ep)** | **37 (1.85/ep)** |
| epizody bez wciśnięcia | 8 | 4 | 4 |
| btn3 | 0 | **1** | 1 |
| mediana kroków/wciśnięcie | 175 | 182 | 225 |

Trend jest jednoznacznie dodatni: **+42 % wciśnięć na 1.5M kroków**, o połowę
mniej epizodów pustych i **pierwsze w historii wciśnięcie btn3** — co
domyka argumentację z rozdz. 6.11 (btn3 to niedouczenie, nie geometria).

**Wąskie gardło nie jest tam, gdzie wyglądało.** Mediana tempa (~180 kroków
na wciśnięcie) mieściłaby pięć przycisków w horyzoncie 1000 (5 × 180 = 900);
problemem jest **wariancja** — pojedyncze wciśnięcia zajmują od 27 do ~1200
kroków, więc jeden trudny przycisk zjada cały epizod. Stąd rozkład urywa się
na 4/5 i nigdy nie sięga kompletu.

**Konsekwencja dla uczenia (właściwa przyczyna zastoju):** polityka **nigdy
nie zaliczyła kompletu**, więc nigdy nie doświadczyła ani `COMPLETE_BONUS`
(+100 — największy pojedynczy składnik rewardu, 44 % maksimum epizodu), ani
terminacji po sukcesie. Uczy się wyłącznie z `PRESS_BONUS` (+25), czyli
optymalizuje „wciskaj cokolwiek", a nie „domknij sekwencję". Funkcja wartości
nie ma jak wycenić stanu „zostały 2 przyciski".

**Decyzja:** horyzont treningowy 1000 → **2000** jako kolejny element
kurikulum (`training/configs/ppo_board_6dof_s1_h2000.yaml`, wznowienie
z 12.41M, +4M). Uzasadnienie: przy horyzoncie 2000 ten sam model robi 1.85
zamiast 1.35 wciśnięć na epizod, więc szansa dojścia do kompletu — a tym
samym zamknięcia pętli kredytu na `COMPLETE_BONUS` — rośnie istotnie.
Po opanowaniu sekwencji horyzont wraca do 1000, żeby protokół ewaluacji
pozostał porównywalny z 7DOF (rozdz. 6.3).

### 6.13. Dlaczego 6DOF uczy się ~9× wolniej — audyt i restart od zera (2026-07-31)

> **KOREKTA (2026-08-02, rozdz. 6.15):** wniosek tego rozdziału — że różnicę
> tłumaczy manipulowalność Yoshikawy — jest **nieuprawniony**. Baza 7DOF
> była mierzona przy chwytaku swobodnym, 6DOF przy wymuszonym zamkniętym,
> a obie przy błędnej geometrii (baza o 0.91 m za wysoko, tablica na 1.20 m
> zamiast 60 cm, brak obrotu 45°). Porównanie nigdy nie było kontrolowane.

Po 16M kroków (~30 h) przebieg 6DOF osiągnął **39 sukcesów na 680 epizodów
(5.7 %)**, i to przy warunkach ułatwionych względem 7DOF (próg 10 mm zamiast
12, horyzont 2000 zamiast 1000), podczas gdy 7DOF w runie v4 miał pierwsze
sukcesy przy **1.55M** i 50–70 % przy 2.8–3M. Przeprowadzono pełny audyt
różnicy.

**Wykluczone (zmierzone, bez różnic):**

- hiperparametry PPO, domain randomization, szum percepcji — identyczne
  z `ppo_board.yaml` (run v4),
- kontroler: ten sam `OSC_POSE`, kp 150, damping_ratio 1,
- fizyka członów: masa ramienia 6.78 kg (6DOF) vs 7.82 kg (7DOF), aktuatory
  skonfigurowane tak samo,
- **osobliwości**: mediana liczby warunkowej jakobianu w pozach dociskowych
  14.8 (6DOF) vs 14.1 (7DOF) — praktycznie identyczna,
- **rozdęte `std` polityki** (do 30 przy 15.8M) — sprawdzono i **odrzucono
  jako przyczynę**: działający model 7DOF v23_12M (90/100) ma `std` średnio
  13.7, w tym 28.1 na jednej osi. Wysokie `std` jest normą w tym setupie
  (reward normalizowany /225 → małe gradienty, człon entropii dominuje),
  a ewaluacja jest deterministyczna, więc `std` jej nie dotyczy.

**Realna różnica kinematyczna:** manipulowalność Yoshikawy
`sqrt(det(J·Jᵀ))` w pozach przy tablicy wynosi średnio **0.0248 (6DOF)
vs 0.0423 (7DOF)** — 6DOF ma **1.7× mniejszą zdolność manipulacyjną**.
Dochodzi brak redundancji: 7DOF ma dla każdej pozy chwytaka ciągłą
1-wymiarową rodzinę konfiguracji (nullspace), 6DOF — kilka dyskretnych.
Dla eksploracji RL to fundamentalna różnica: 7DOF może dojść do celu
„z dowolnej strony", korygując konfigurację bez zmiany pozy narzędzia.
To wyjaśnia różnicę rzędu 2–3×, **nie 9×**.

**Reszta różnicy to skażenie przebiegu (błąd metodyczny):**

| zakres kroków | co było nie tak |
|---|---|
| 0 – 5.2M (33 %) | trening z **fantomowym kontaktem** base×shoulder (rozdz. 6.9) — polityka uczyła się kompensować siłę, która nie istnieje |
| @3.9M | doklejony `depress_shaping_k` |
| @7.7M | zmiana progu 12 → 10 mm |
| @12.4M | zmiana horyzontu 1000 → 2000 |

Każda poprawka trafiała na **już zdeformowaną politykę**, a środowisko było
niestacjonarne przez cały przebieg — najgorszy możliwy scenariusz dla RL
(funkcja wartości uczy się rozkładu, który zaraz przestaje obowiązywać).
Run v4 na 7DOF miał od kroku 0 warunki stałe.

**Decyzja: czysty scratch** (`training/configs/ppo_board_6dof_s1_clean.yaml`,
`logs/board_6dof/s1_clean/`) — wszystkie ustalenia od kroku 0 i stacjonarnie:
naprawiony XML, próg 10 mm, shaping głębokości, horyzont 2000, reward
v2-like, bramka OFF; hiperparametry celowo **niezmienione** względem v4
(lekcja: nie zmieniać wielu rzeczy naraz). Przebieg skażony zachowany
w `logs/board_6dof/s1_contaminated/` jako materiał porównawczy.

**Oczekiwanie:** 7DOF potrzebował 1.55M kroków do pierwszych sukcesów;
przy 1.7× gorszej manipulowalności szacunek dla 6DOF to 3–6M. Jeśli do
~6M nie będzie sukcesów, hipoteza „różnica jest tylko kinematyczna" upada
i trzeba szukać dalej.

### 6.14. Czysty scratch 6DOF domknięty (12M): diagnoza potwierdzona, ale sufit bez zmian (2026-08-02)

> **KOREKTA (rozdz. 6.15):** teza tego rozdziału, że problem btn3 to „realna
> własność zadania dla tej kinematyki", jest **fałszywa** — 7DOF trafia btn3
> 95/100 pod tym samym kodem. btn3 był najdalszym przyciskiem wyłącznie
> z powodu błędnej geometrii. Wyniki poniżej opisują zadanie, które nie
> odpowiada ani konfiguracji 7DOF, ani regulaminowi zawodów.

Przebieg `s1_clean` doprowadzony do **12.0M kroków** (ostatnie 5.7M w jednym
biegu, 15.2 h, średnio 104 kroki/s wliczając narzut ewaluacji co 50k).
Zakończony czysto, bez błędów; modele: `logs/board_6dof/s1_clean/best_model/`
(punkt 11.25) i `ppo_board_final.zip`.

**Ewaluacja końcowa** (`compare_models.py`, 100 epizodów parowanych,
horyzont 2000, próg 10 mm — czyli warunki *treningowe*, ułatwione):

| metryka | `best_model` | `ppo_board_final` |
|---|---|---|
| sukcesy (5/5) | **3/100** | **3/100** |
| średni czas sukcesu | 1382 kroków (69 s) | 1599 kroków (80 s) |
| epizody z kolizją | 92/100 | 86/100 |
| śr. kroków kolizji/epizod | 247.9 | 86.0 |
| epizody z misuse | 58/100 | 52/100 |
| stuck | 17/100 | 12/100 |
| porażki wg wciśniętych (0..4) | [20, 33, 25, 11, 8] | [27, 28, 19, 14, 9] |

Różnica best vs final nieistotna (Fisher exact, **p = 1.00**) — model
„najlepszy" wg rewardu eval nie jest lepszy zadaniowo, co przy 3 sukcesach
na 100 jest spodziewane (reward eval jest zdominowany przez shaping, nie
przez sukcesy).

**Wciśnięcia per przycisk** (kluczowa diagnoza):

| | btn0 | btn1 | btn2 | btn3 | btn4 |
|---|---|---|---|---|---|
| `best` | 38 | 32 | 42 | **15** | 36 |
| `final` | 40 | 34 | 41 | **8** | 36 |

Średnio **1.63 wciśnięcia na epizod** (best) przy 5 wymaganych. Problem
btn3 z rozdz. 6.11 **powtórzył się w przebiegu całkowicie czystym** —
to zamyka sprawę: nie był artefaktem skażenia, tylko realną własnością
zadania dla tej kinematyki.

**Porównanie trzech przebiegów:**

| przebieg | pierwszy sukces | wynik końcowy |
|---|---|---|
| 7DOF, run v4 → v2.3 | 1.55M | **90/100** |
| 6DOF skażony (`s1_contaminated`) | 13.46M | 39/680 = 5.7 % @16M |
| 6DOF czysty (`s1_clean`) | **5.7M** | 3/100 @12M |

**Wniosek — hipoteza potwierdzona tylko połowicznie.** Restart potwierdził
diagnozę z rozdz. 6.13 co do **tempa uczenia**: pierwszy sukces przyszedł
2.4× wcześniej (5.7M vs 13.46M) i trafił w prognozowane okno 3–6M. Ale
**sufit się nie podniósł** — oba przebiegi 6DOF stoją w okolicy 3–6 %,
przy czym wskaźnik z callbacku eval (34/1140 = 3.0 % na 6.35–12M, w tym
5.2 % w ostatnim 1M) potwierdza plateau od ~8.4M. Skażenie wyjaśniało
więc, *jak wolno* polityka się uczyła, a nie *dlaczego zatrzymuje się
tak nisko*. Przyczyna plateau pozostaje otwarta.

**Nowa obserwacja: kolizyjność.** 92/100 epizodów z kontaktem z płytą,
średnio 248 kroków na epizod (12 % długości epizodu). Przy `collision_cost
0.5` (reward v2-like, łagodny — świadomy wybór etapu 1) polityka nie ma
bodźca, by tablicy unikać; w skrajnym epizodzie 1485/2000 kroków to
docisk do płyty przy `||da||` 0.39 (polityka „parkuje" na tablicy).
Dla sim2real jest to istotniejsze niż sam success rate — patrz nagrania.

**Nagrania** (`docs/video/`, odtwarzalne przez `--seed`, parowanie jak
w `compare_models.py`):

- `6dof_s1_clean_12M_sukces.mp4` — seed 1307, pełne 5/5 w 1284 krokach, 2 kroki kolizji,
- `6dof_s1_clean_12M_porazka_3z5.mp4` — seed 1238, typowa porażka 3/5,
- `6dof_s1_clean_12M_patologia_docisk.mp4` — seed 1235, docisk do tablicy (reward −33.2).

**Decyzja: etap 2 (fine-tune na warunki docelowe) WSTRZYMANY.** Config
`ppo_board_6dof_s2_target.yaml` (horyzont 1000, próg 12 mm) jest gotowy,
ale odpalanie go z polityki o 1.63 wciśnięcia/epizod oznaczałoby
jednoczesne **skrócenie budżetu epizodu o połowę** i **podniesienie progu
o 20 %** — przy prostej ekstrapolacji ok. 0.8 wciśnięcia/epizod, czyli
sukces bliski zeru. Warunkiem wejścia w etap 2 jest najpierw podniesienie
sufitu etapu 1.

### 6.15. Audyt geometrii i chwytaka: cztery usterki, które unieważniają wnioski 6.13–6.14 (2026-08-02)

Po domknięciu etapu 1 na 3/100 (rozdz. 6.14) przeprowadzono audyt środowiska
względem commitów, w których 7DOF **nauczył się zadania od zera**
(`94d3816`, `9f8b4dd` — reward v2, pierwsze 5/5 przy 1.55M).

**Co się zgadzało** (sprawdzone linia po linii): stałe rewardu
(`SHAPING_K` 10, `PRESS_BONUS` 25, `COMPLETE_BONUS` 100, `DENSE_COEF` 0.05,
`ACTION_COST` 0.01, `REWARD_NORM` 225), hiperparametry PPO
(`s1_clean.yaml` vs `ppo_board.yaml@9f8b4dd`), shaping potencjałowy,
terminacja po sukcesie, `_check_board_collision` (różnica: spacja).
Wykrywanie kolizji jest poprawne — kapturki to `btn*_cap_collision`,
filtr `"board" in g` łapie wyłącznie `omniboard_board_collision`, więc
wciskanie nigdy nie było liczone jako kolizja. Test regresji: model 7DOF
`ppo_board_v23_12M` daje pod obecnym kodem **87/100** (udokumentowane 90/100),
z rozbiciem per przycisk 93/96/96/**95**/93 — czyli **btn3 nie jest trudny
geometrycznie**, wbrew tezie z 6.11 i 6.14.

**Usterka 1 — `force_gripper_closed` narzucony od kroku 0.** Wprowadzony
commitem `7332a99` („now it gets points only for clicking with closed
gripper"), czyli **już po** udanym scratchu v4 — ten uczył się z chwytakiem
w pełni sterowanym przez politykę (w `9f8b4dd` nie ma metody `step()`).
`envs/button_board.py` nadpisuje `action[-1] = 1.0` co krok, a
`train_board.py` **nie przekazywał tej flagi**, więc żaden YAML nie mógł jej
wyłączyć. Pomiar A/B na modelu v4, 40 epizodów parowanych:

| warunki | sukcesy | wciśnięcia/epizod |
|---|---|---|
| chwytak wolny (jak w treningu v4) | **33/40 (82 %)** | 4.67 |
| chwytak wymuszony zamknięty | **20/40 (50 %)** | 3.60 |

Samo zaspawanie chwytaka odbiera **32 punkty procentowe** gotowej polityce;
średnia komenda chwytaka −0.263 pokazuje, że polityka aktywnie chce go
otwierać. U 7DOF restrykcję nałożono **fine-tunem na politykę, która już
umiała zadanie** (v4 → v2.1 → `7332a99` → v2.3, 90/100). Bieg 6DOF nałożył
ją od kroku 0, pomijając fazę łatwej eksploracji.

**Usterka 2 — baza ramienia o 0.91 m za wysoko.** `KINOVA_BASE_Z = 0.65`
z komentarzem „Spot body (~0.5m) + arm mount (~0.15m)" było podawane wprost
do `set_base_xpos()`, które ustawia **spód cokołu**, a nie bazę ramienia.
Cokół `RethinkMount` ma `bottom_offset = −0.912 m`, więc:

| `KINOVA_BASE_Z` | faktyczne `robot0_base` |
|---|---|
| 0.00 | 0.912 m |
| 0.65 | **1.562 m** |

Ramię stało na 1.56 m — nie na Spocie, tylko na Spocie postawionym na stole.

**Usterka 3 — tablica na 1.20 m zamiast 60 cm.** Regulamin RoboCup Rescue
League: „Omni may be mounted on the wall at 60 cm or floor"; 60 cm to
wysokość środka tablicy. W symulacji `BOARD_Z = 1.20`.

**Usterka 4 — brak obrotu tablicy o 45°.** W zawodach omniboard montuje się
„na rogu": przyciski leżą na przekątnych (góra-lewo / góra-prawo /
dół-lewo / dół-prawo), a nie w pionie i poziomie. `BUTTON_FACES` miało
układ krzyżowy.

**Skutek łączny dla geometrii** (przyciski względem bazy ramienia):

| | zamierzone | było w symulacji | po naprawie |
|---|---|---|---|
| baza ramienia | 0.65 m | 1.562 m | 0.65 m |
| środek tablicy | 0.60 m | 1.241 m | 0.60 m |
| przyciski wzgl. bazy | opasują bazę | **−0.48 … −0.16 m** | −0.16 … +0.06 m |
| odległość btn3 od bazy | — | **0.638 m (najdalej)** | 0.466 m |
| rozrzut odległości | — | 0.454–0.638 m | 0.370–0.466 m |

W starej geometrii btn3 był najdalszym przyciskiem **i** leżał na dolnym
ścięciu (ścianka patrzy w dół), więc ramię musiało sięgnąć najdalej w dół,
wejść pod przycisk i pchać do góry — najbardziej wykręcona poza z piątki.
Diagnostyka polityki 6DOF to potwierdza: przy btn3 podchodziła **najbliżej**
(mediana 14.3 mm) i spędzała **najwięcej kroków** (630), osiągając 9.0 mm
przy progu 10 mm. Po obrocie o 45° i sprowadzeniu tablicy na 60 cm
odległości są wyrównane (0.37–0.47 m), a btn3 przestaje być odstający.

To także tłumaczy, skąd wzięło się obniżenie progu 12 → 10 mm (rozdz. 6.10):
**nie było kurikulum, tylko obejściem złej geometrii.**

**Naprawa** (`envs/button_board.py`):

- `KINOVA_BASE_Z` jest teraz wysokością **bezwzględną**; `_load_model`
  koryguje ją o `robot_model.bottom_offset` (nie zaszyte 0.912),
- `BOARD_Z = 0.60`, `BOARD_ROLL_DEG = 45.0`; `BUTTON_FACES` liczone przez
  `_rolled_button_faces()`, a bryła tablicy dostaje **ten sam** obrót
  (`R = Rx(roll) @ Ry(−90°)`), więc przyciski zawsze leżą na swoich ściankach,
- `force_gripper_closed` i `base_x_offset` przekazywane z YAML,
- `base_x_offset` domyślnie **−0.10**: po sprowadzeniu tablicy poza domowa
  ramienia (TCP na x≈0.43) trafiała dokładnie w kapturki (x=0.423) i epizod
  startował z penetracją przycisków — polityka dostawała kontakt w kroku 0.
  Przy −0.10 jest **zero kontaktów startowych** dla 6DOF i 7DOF przy pełnej
  randomizacji (12 resetów),
- kamera `board_view` przekadrowana pod nową scenę.

Weryfikacja: baza 0.65, tablica 0.60, przyciski na przekątnych, zero
kontaktów przy resecie, wymiar obserwacji bez zmian (120), trening startuje
(266 kroków/s). Zrzuty: `docs/img/env_geometry_before.png`,
`env_geometry_after.png`, `env_geometry_fixed.png`.

**Konsekwencje dla wcześniejszych wniosków:**

- **Wszystkie wyniki 6DOF (6.8–6.14) dotyczą innego zadania** niż to, na
  którym mierzony był 7DOF, i innego niż zawody. Modele 6DOF wczytają się
  (obs 120 bez zmian), ale są bezwartościowe.
- **Teza z 6.13, że różnicę tłumaczy manipulowalność Yoshikawy, jest
  nieuprawniona** — baza 7DOF była mierzona przy wolnym chwytaku, a obie
  konfiguracje przy złej geometrii. Porównanie nigdy nie było kontrolowane.
- **Teza z 6.11/6.14, że btn3 to własność kinematyki 6DOF, jest fałszywa** —
  7DOF trafia btn3 95/100 pod tym samym kodem.

**Dalej:** `training/configs/ppo_board_6dof_v4like.yaml` — replikacja
warunków v4 (chwytak wolny, bramka OFF, próg **12 mm** nominalny, horyzont
1000, kolizja 0.5, bez shapingu głębokości) na poprawionej geometrii.
To jest kontrolowany eksperyment, którego brakowało w 6.13.

### 6.16. Bieg `v4like` na poprawionej geometrii: geometria naprawiona, wąskim gardłem próg głębokości (2026-08-03)

Pierwszy scratch po naprawach z rozdz. 6.15
(`ppo_board_6dof_v4like.yaml`): 6.01M kroków, 11.6 h, 144 kroki/s, czysty
exit. Warunki v4 (chwytak wolny, bramka OFF, **próg 12 mm**, horyzont 1000,
kolizja 0.5, `depress_shaping_k = 0`) na robocie 6DOF.

**Ewaluacja końcowa** (50 epizodów parowanych, horyzont 1000, chwytak wolny):

| | sukcesy | wciśnięcia/ep | kroków kolizji/ep | btn0/1/2/3/4 |
|---|---|---|---|---|
| `best`, próg 12 mm | 0/50 | 0.88 | 36.7 | 11/12/9/**2**/10 |
| `final`, próg 12 mm | 0/50 | 1.18 | 36.9 | 14/14/12/**10**/9 |
| `best`, próg 10 mm (kontrola) | 1/50 | 1.48 | 45.7 | 17/16/16/**9**/16 |

**Co naprawa geometrii faktycznie dała** (porównanie ze starym `s1_clean`
@12M, próg 10 mm, horyzont 2000 — 3/100, 1.63 wciśnięcia/ep, 248 kroków
kolizji/ep):

- **Problem btn3 zniknął.** Model finalny: 14/14/12/**10**/9 — btn3 w normie.
  Stary bieg: 15/100 przy 32–42 dla reszty. Potwierdza rozdz. 6.15: btn3
  był artefaktem geometrii, nie kinematyki 6DOF.
- **Kolizyjność ~7× niższa**: 36.7 vs 248 kroków/epizod. Dla sim2real
  istotniejsze niż sam success rate.
- **Efektywność wciśnięć ~1.8× wyższa**: 1.48 wciśnięcia na 1000 kroków
  wobec 1.63 na 2000 kroków (0.82/1000) — przy **połowie budżetu**
  treningowego (6M vs 12M).

**Czego nie osiągnięto i dlaczego.** Sukcesów ~0 (0/50 przy 12 mm, 1/50 przy
10 mm). Pomiary w trakcie biegu (12 epizodów, model `best`):

| krok | mediana głębokości | min. dystans TCP–cel | wciśnięcia/ep |
|---|---|---|---|
| 3M | 7.7 mm | 11.8 mm | 0.13 |
| 4M | 7.6 mm | 9.6 mm | 0.25 |

Polityka **dolatuje i dociska**, ale zatrzymuje się ~4 mm pod progiem, a
głębokość **nie drgnęła** między 3M a 4M (max 11.6 mm, 0/12 epizodów ≥12 mm).
Przy `depress_shaping_k = 0` próg jest funkcją **schodkową**: na odcinku
7.6 → 12 mm nie ma żadnego gradientu, więc nie ma po czym się wspinać.
7DOF radził sobie bez shapingu, bo dociskał 12–16 mm.

**Uwaga metodyczna — `best_model` bywa mylący.** Model finalny jest
**lepszy zadaniowo** od `best` (1.18 vs 0.88 wciśnięcia/ep; btn3 10 vs 2),
bo `EvalCallback` wybiera po rewardzie eval, który przy zerze sukcesów jest
zdominowany przez shaping i kary, nie przez wciśnięcia. Przy ocenie biegów
bez sukcesów trzeba patrzeć na oba modele.

**Pułapka pomiarowa (naprawiona).** `compare_models.py` i `watch_board.py`
budowały env bez `force_gripper_closed`, więc modele trenowane z **wolnym**
chwytakiem oceniano z **zaspawanym** — pomiar @2M dawał 0.07 zamiast 0.27
wciśnięcia/ep. Dodano flagę `--free-gripper`; ewaluacja wewnątrz treningu
(`EvalCallback`) była poprawna od początku, bo dziedziczy `env_kwargs`.

**Test wykonalności docisku — NIEROZSTRZYGAJĄCY.** Skryptowany docisk
(P-regulator na oś przycisku) osiągnął w nowej geometrii 3.8–6.3 mm przy
chwytaku zamkniętym i ~0 przy otwartym. Próba kontrolna **na starej
geometrii dała 3.4–9.9 mm**, czyli skrypt jest równie słaby w obu — nie
mówi nic o wykonalności 12 mm (ta sama pułapka co w rozdz. 6.10). Jedyny
użyteczny wniosek: **przy otwartym chwytaku TCP dochodzi na 1.3 mm do
kapturka bez żadnego wciśnięcia** — palce omijają grzybek bokami.

**Dalej:** `ppo_board_6dof_depress.yaml` — ten sam config z **jedną**
zmienioną zmienną (`depress_shaping_k` 0 → 400, wartość z `s1_clean`),
budżet 6M, żeby porównanie było kontrolowane. Shaping jest potencjałowy
i teleskopuje się; pełne 12 mm daje 4.8 wobec `PRESS_BONUS` 25 (~19 %).

> **Realizacja:** zamiast scratchu wykonano **fine-tune** z modelu `v4like`
> @6M (rozdz. 6.17). Polityka miała już to, co trudne (dolot, unikanie
> kolizji, docisk do 7.6 mm) — brakowało jej przyrostu na **jednej**
> wielkości i właśnie na nią shaping daje gradient. Scratch okazał się
> zbędny; `ppo_board_6dof_depress.yaml` pozostaje nieużyty.

### 6.17. Shaping głębokości domyka zadanie 6DOF: 0 → 80/100 (2026-08-04)

Bieg `ppo_board_6dof_depress_ft.yaml` — **jedyna** zmiana względem `v4like`
to `depress_shaping_k` 0 → 400. Wznowienie z `v4like/ppo_board_final.zip`
(nie `best_model` — patrz uwaga metodyczna w 6.16), w dwóch etapach:

| etap | zakres | budżet | czas |
|---|---|---|---|
| 1 | 6.00M → 8.29M | 2.29M kroków | ~4.5 h |
| 2 | 8.21M → 14.23M | 6.01M kroków | 11 h 07 min |

Etap 1 przerwano na życzenie prowadzącego; etap 2 wznowiono z checkpointu
`ppo_board_8212928_steps.zip` (utrata 77k kroków — checkpointy co 100k).

**Krzywa uczenia** (`EvalCallback`, 10 epizodów co 50k kroków; sukces =
epizod krótszy od horyzontu, bo środowisko terminuje po komplecie):

| odcinek | sukces | reward eval | dł. epizodu |
|---|---|---|---|
| 6.0–6.5M | 0 % | −0.029 | 1000 |
| 6.5–7.0M | 3.0 % | +0.073 | 997 |
| 7.0–7.5M | 9.0 % | +0.203 | 988 |
| 7.5–8.0M | — | +0.215 | 983 |
| 8.0–8.5M | 16.0 % | +0.250 | 978 |
| 8.5–9.0M | 27.0 % | +0.389 | 944 |
| 9.0–9.5M | 42.0 % | +0.473 | 893 |
| 9.5–10.0M | 49.0 % | +0.572 | 854 |
| 10.0–10.5M | 47.0 % | +0.538 | 860 |
| 10.5–11.0M | 64.0 % | +0.691 | 801 |
| 11.0–11.5M | 63.0 % | +0.594 | 833 |
| 11.5–12.0M | 68.0 % | +0.761 | 771 |
| 12.0–12.5M | 71.0 % | +0.731 | 725 |
| 12.5–13.0M | 73.0 % | +0.738 | 739 |
| 13.0–13.5M | 77.0 % | +0.826 | 699 |
| 13.5–14.0M | 78.0 % | +0.833 | 695 |
| 14.0–14.21M | 70.0 % | +0.699 | 749 |

**Diagnoza z 6.16 potwierdzona.** Mediana głębokości ruszyła natychmiast po
włączeniu shapingu: 7.6 mm → **10.9 mm po 0.5M** → **11.4 mm po 1M**.
Pierwszy sukces padł @6.76M — `v4like` nie miał **żadnego** przez całe 6M
przy identycznych warunkach. Próg 12 mm rzeczywiście był funkcją schodkową,
a nie granicą wykonalności.

**Ewaluacja końcowa** (100 epizodów parowanych, horyzont 1000,
`--free-gripper`, `--robot Kinova3_6DOF`):

| model | sukcesy | czas sukcesu | kolizje/ep | misuse | btn0/1/2/3/4 |
|---|---|---|---|---|---|
| **`final` @14.23M** | **80/100** | 594 kroków (29.7 s) | 41.6 | 16/100 | 96/91/89/**90**/90 |
| `best_model` | 80/100 | 628 kroków (31.4 s) | 45.9 | 14/100 | 94/91/88/**88**/96 |
| `etap1` @8.21M | 12/100 | 805 kroków (40.3 s) | 37.2 | 16/100 | 60/56/61/**47**/58 |

Różnica `final` vs `etap1` jest przytłaczająca: Fisher **p = 2.4·10⁻²³**,
parowo 71 epizodów wygranych wyłącznie przez `final` wobec 3 przez `etap1`.
`final` vs `best` są **nierozróżnialne** (80 vs 80, p = 1, po 16 epizodów
na wyłączność każdy) — do dalszych prac przyjęto `final`, bo jest o 34 kroki
szybszy i mniej kolizyjny. Przy niezerowym wskaźniku sukcesu `best_model`
przestał być mylący, w odróżnieniu od sytuacji z rozdz. 6.16.

**btn3 ma 90/100 — tyle samo co u 7DOF.** Rozkład per przycisk jest płaski
(96/91/89/90/90). To ostatecznie domyka wątek z rozdz. 6.8–6.14: btn3 nigdy
nie był trudny kinematycznie, był tylko najdalszym przyciskiem w zepsutej
geometrii. Obniżenie progu 12 → 10 mm z rozdz. 6.10 było obejściem usterki,
nie kurikulum — cały bieg przeprowadzono na **nominalnym progu 12 mm**.

**Koszt: kolizyjność.** 99/100 epizodów ma kontakt z płytą, średnio 41.6
kroku, i wskaźnik **wzrósł** względem etapu 1 (37.2) oraz `v4like` (36.7).
Polityka dociska skuteczniej częściowo przez **opieranie się o tablicę** —
to bezpośredni skutek uboczny `depress_shaping_k = 400`, który nagradza
głębokość, a kara za kolizję wynosi tylko 0.5. Dla sim2real jest to istotne:
kontakt siłowy z tablicą jest na sprzęcie groźniejszy niż niższy wynik.
**Mitygacja po stronie nadzorcy** (limit siły/prędkości w `SafetyLayer`,
rozdz. 5.1), nie rewardem — próby naprawiania zachowania rewardem trzykrotnie
zawiodły przy 7DOF (rozdz. 6.5–6.6).

**Wideo** (`docs/video/`, seedy sparowane z `compare_models.py`, epizody
odtwarzają się co do kroku):

- `6dof_depress_ft_sukces_szybki.mp4` — seed 1287, komplet w 280 krokach, 8 kroków kolizji
- `6dof_depress_ft_porazka_4z5.mp4` — seed 1285, 4/5 przycisków, typowa porażka
- `6dof_depress_ft_docisk_kolizyjny.mp4` — seed 1292, sukces w 988 krokach przy 97 krokach kolizji (ilustracja patologii opisanej wyżej)

**Bilans 6DOF vs 7DOF.** ~~80/100 wobec 90/100 na plateau u 7DOF; pozostałe
~10 pp to realny koszt utraty jednego stopnia swobody.~~ **TO PORÓWNANIE JEST
NIEWAŻNE — patrz rozdz. 6.22.** Liczba 90/100 pochodzi ze **starej geometrii**
(model 7DOF z 19.07, poprawki geometrii z 02.08), więc zestawiano dwa różne
zadania. Przemierzony na obecnej geometrii ten sam model daje **0/100**.
Wniosek, który pozostaje w mocy: wyniki 6DOF **nie są** dowodem nieuczalności
zadania, jak sugerowały nieaktualne rozdziały 6.8–6.14. Model 6DOF jest gotowy
do wpięcia w nadzorcę sim2real.

**Uwaga narzędziowa.** `EvalCallback` przy wznowieniu **nadpisuje**
`eval/evaluations.npz` — dane per epizod z etapu 1 przepadły. Ocalała
natomiast krzywa w TensorBoardzie: SB3 przy `reset_num_timesteps=False`
kontynuuje w **tym samym** katalogu (`ppo_board_6dof_depress_ft_0`), więc
oba etapy leżą tam jako dwa pliki zdarzeń i składają się w ciągłą krzywą
6M → 14.2M. Przy wznowieniach warto kopiować `evaluations.npz` przed startem.

### 6.18. Zacieśnianie do pełnego rewardu v2.3: kolizyjność 4.7× w dół, ale bieg kolapsuje po ~2.5M kroków (2026-08-05)

Etap docelowy 6DOF: przejście z warunków „uczących" (wolny chwytak, łagodna
kara kolizyjna) na **protokół sprzętowy** = domyślną konfigurację
`ButtonBoardEnv`, czyli reward v2.3, ten sam, na którym mierzony był 7DOF.
Config `training/configs/ppo_board_6dof_s2.yaml`, fine-tune z modelu
z rozdz. 6.17 (`depress_ft/ppo_board_final.zip`, 80/100), budżet 6M kroków
dodatkowych (14.23M → 20.24M, 11 h 17 min, 120 ewaluacji).

**Cztery zmiany naraz** (świadomie, jako osobny stacjonarny etap):

| parametr | 6.17 | ten etap | po co |
|---|---|---|---|
| `force_gripper_closed` | false | **true** | na sprzęcie wciskamy czubkiem zamkniętych palców; otwarty chwytak uderzałby kamerą |
| `finger_press_gate` | false | **true** | odpowiednik detektora wciśnięcia w TCP u nadzorcy (rozdz. 5.1) |
| `collision_cost` | 0.5 | **5.0** | opieranie się o tablicę z 6.17 |
| `action_rate_cost` | 0.0 | **0.02** | v2.1+ |

`depress_shaping_k = 400` **zostawiono** (v2.3 u 7DOF miał 0). Uzasadnienie:
przy zamkniętym chwytaku zmienia się geometria kontaktu, więc głębokość trzeba
przeuczyć, a shaping jest potencjałowy i teleskopuje się, więc nie zmienia
polityki optymalnej. Jak się okazało, **to była błędna decyzja** — patrz niżej.

**Ryzyko policzone przed startem.** Model startowy kolidował w 99/100 epizodów
po 41.6 kroku, więc przy `collision_cost = 5.0` kara startowa to ~208 na epizod
wobec `REWARD_NORM` 225 — kara równa całej nagrodzie zadaniowej. To układ,
który u 7DOF zapadł się w v2.2 (rozdz. 6.3). Różnica na korzyść: tam karany był
także kontakt **chwytaka** (`finger_misuse_cost`), tu misuse = 0.

**Przebieg: wzrost do 16.5M, potem monotoniczny kolaps.**

| okno | sukces | reward eval | ep_len |
|---|---|---|---|
| 14.2–14.7M | 22 % | −1.16 | 936 |
| 14.7–15.2M | 35 % | −0.80 | 877 |
| 15.2–15.7M | 41 % | −0.49 | 861 |
| 15.7–16.2M | 55 % | −0.37 | 815 |
| **16.2–16.7M** | **59 %** | **−0.21** | **809** |
| 16.7–17.2M | 40 % | −0.57 | 852 |
| 17.2–17.7M | 51 % | −0.34 | 832 |
| 17.7–18.2M | 40 % | −0.54 | 891 |
| 18.2–18.7M | 30 % | −0.97 | 915 |
| 18.7–19.2M | 18 % | −1.15 | 962 |
| 19.2–19.7M | 4 % | −1.47 | 995 |
| 19.7–20.2M | 2 % | −1.22 | 998 |

Szok po przełączeniu configu (80 % → ~22 %) był zgodny z przewidywaniem
(A/B na v4: samo zamknięcie chwytaka ≈ −32 pp). Odbudowa do 59 % zajęła 2.3M
kroków. Potem polityka **zjechała do zera** — nie na plateau, tylko w dół,
monotonicznie przez ostatnie 3.5M kroków.

**Ewaluacja 100 sparowanych epizodów** (`compare_models.py`, próg nominalny
12 mm, chwytak zamknięty — tak jak model był uczony, więc **bez**
`--free-gripper`; `logs/board_6dof/s2/eval_100ep.csv`):

| model | sukces | kroki kolizji/ep | epizody z kolizją | misuse/ep | czas sukcesu |
|---|---|---|---|---|---|
| `ckpt15.93M` | 54/100 | 19.1 | 84/100 | 1.0 | 658 kr. |
| **`ckpt16.43M` = `best_model`** | **52/100** | **8.8** | 83/100 | **0.4** | 665 kr. |
| `final` @20.24M | 1/100 | 26.1 | 81/100 | 1.4 | — |

`best_model.zip` okazał się **co do epizodu tym samym modelem** co checkpoint
16.43M (zgodność w 100/100 epizodów) — tym razem `EvalCallback` trafił dokładnie
w optimum, inaczej niż w sytuacji z rozdz. 6.16, gdzie przy zerze sukcesów
wybierał po rewardzie zdominowanym przez shaping.

54 vs 52 to remis (Fisher p = 0.89, pary rozbieżne 29:27) — rozstrzyga
kolizyjność, stąd wybór `ckpt16.43M`. Względem `final`: p = 1.8·10⁻¹⁸,
pary 51:0.

**Wynik właściwy tego etapu — kolizyjność.** Wobec modelu z 6.17:

| metryka | 6.17 (wolny chwytak) | 6.18 `best` | zmiana |
|---|---|---|---|
| kroki kolizji / epizod | 41.6 | **8.8** | **4.7× mniej** |
| epizody z kolizją | 99/100 | 83/100 | −16 pp |
| misuse ramienia / epizod | 1.4 | 0.4 | 3.5× mniej |
| sukces | 80/100 | 52/100 | −28 pp |

Czyli `collision_cost = 5.0` **zadziałał** — pierwszy raz w całym projekcie
zachowanie kontaktowe poprawiło się rewardem, a nie mimo niego. Cena to 28 pp
sukcesu, z czego ~32 pp przypisywało A/B samemu zamknięciu chwytaka; realnie
więc bramka palcowa i kary kolizyjne wyszły blisko zera netto, a zysk siedzi
w kolizyjności. To jest **pierwszy model 6DOF nadający się na sprzęt**:
wciska zamkniętymi palcami, przez bramkę palcową, prawie nie opierając się
o tablicę.

**Diagnoza kolapsu.** Skośna para `depress_shaping_k = 400` (nagradza wciskanie
głębiej, czyli mocniejszy nacisk) wobec `collision_cost = 5.0` (karze kontakt)
jest **niestabilna w długim biegu**: przez ~2.5M kroków wygrywa zadanie, potem
kara. Epizody katastroficzne (reward < −10, czyli zacięcie oparte o tablicę
przez cały horyzont) narastały: 1/200 w oknie 15–16M, 2/90 w 18.2M+, łącznie
9/1200. Mechanizm jest ten sam co w v2.2 u 7DOF (rozdz. 6.3), tylko **opóźniony
o 2.5M kroków** przez shaping głębokości — dlatego kryterium stopu ustawione
na 2M kroków go nie wykryło. Wniosek metodologiczny: **kryterium stopu
sprawdzające tylko, czy polityka się podnosi, nie chroni przed późną
degradacją** — trzeba patrzeć też na trend malejący po szczycie.

**Propozycje na kolejne podejście** (niewykonane, wymaga osobnego biegu):

1. **wygaszać `depress_shaping_k`** po odzyskaniu głębokości (np. 400 → 0
   liniowo między 1M a 2M kroków fine-tune'u) — shaping jest potrzebny do
   przeuczenia docisku zamkniętymi palcami, ale po tym już tylko konkuruje
   z karą kolizyjną;
2. albo `collision_cost = 2.0` zamiast 5.0 — łagodniejszy gradient, kara ~83/225
   zamiast 208/225;
3. w obu wariantach budżet **3M, nie 6M** — szczyt wypadł przy 2.3M kroków
   dodatkowych, reszta biegu tylko psuła model.

**Model do użycia z tego etapu:** `logs/board_6dof/s2/best_model/best_model.zip`
(= checkpoint 16.43M), 52/100, 8.8 kroku kolizji/ep. `ppo_board_final.zip`
z tego biegu jest **bezwartościowy** (1/100) i służy wyłącznie jako
udokumentowanie kolapsu.

**Wideo** (`docs/video/`, model `best_model.zip`, seedy sparowane
z `compare_models.py`, chwytak zamknięty — epizody odtwarzają się co do kroku):

- `6dof_s2_sukces_bez_kolizji.mp4` — seed 1247, komplet 5/5 w **381 krokach
  przy zerze kroków kolizji**. Epizod tego typu w rozdz. 6.17 nie istniał
  (kolizję miało 99/100 epizodów) — to jest efekt, dla którego robiono ten etap
- `6dof_s2_porazka_4z5.mp4` — seed 1241, 4/5 (nietrafiony btn3), 2 kroki kolizji
- `6dof_s2_docisk_71krokow.mp4` — seed 1270, najgorszy epizod pod względem
  kontaktu (71 kroków kolizji, 1/5 wciśnięty) — resztkowa patologia dociskowa
- `6dof_s2_zaciecie_stuck.mp4` — seed 1305, zacięcie wykryte flagą `stuck`
  (4/5 wciśnięte, EEF przestaje się przemieszczać). Zacięcia: 5/100 epizodów

**Znany artefakt:** w jednym epizodzie modelu `final` MuJoCo zgłosiło
`Nan, Inf or huge value in QACC at DOF 5` — niestabilność solvera przy
dociskaniu w okolicy osobliwości. Nie wpływa na policzone metryki (epizod
dobiegł do końca horyzontu), ale to kolejny argument za limitem siły
w `SafetyLayer`.

### 6.19. Obcięcie nagrody za głębokość i fizyczna ściana (2026-08-05)

Dwie zmiany w `envs/button_board.py`, obie wynikające z rozdz. 6.18: pierwsza
usuwa przyczynę kolapsu, druga usuwa lukę sim2real widoczną na nagraniach.

**1. Reward clipping z 1-milimetrowym buforem overtravel.**

Diagnoza z 6.18: `depress_shaping_k` nagradza wciskanie **bez ograniczenia
w głąb**, a `collision_cost` karze wynikający z tego kontakt. Oba człony ciągną
w przeciwne strony i po ~2.5M kroków wygrywa kara — polityka przestaje dotykać
tablicy w ogóle.

Rozwiązanie: głębokość wchodząca do shapingu jest **obcinana** na
`activation_depth + depress_clip_buffer` (domyślnie 12 + 1 = 13 mm). Za
wciskanie głębiej polityka nie dostaje już nic, a nadal płaci karę kolizyjną,
więc konflikt znika — optimum przesuwa się na „wciśnij do aktywacji i puść".

Nowy parametr konstruktora `depress_clip_buffer = 0.001`, przekazywany z YAML
(`train_board.py`). Przy `depress_shaping_k = 0` jest to no-op, więc **domyślna
konfiguracja v2.3 się nie zmienia** i wyniki 7DOF pozostają porównywalne.

Człon nadal **teleskopuje się**: obcięta głębokość jest funkcją samego stanu,
więc shaping pozostaje potencjałowy i nie zmienia polityki optymalnej — obcięcie
zmienia tylko to, gdzie gradient się kończy.

Weryfikacja liczbowa (przejazd 0 → 25 mm, `depress_shaping_k = 400`):

| bufor | suma shapingu | odpowiada |
|---|---|---|
| 1 mm | **5.20** | 400 × 0.013 = próg + bufor |
| brak (kontrola) | 10.00 | 400 × 0.025 = pełny przejazd |

**2. Ściana `wall_panel` staje się obiektem fizycznym.**

Na nagraniach z 6.18 widać, że manipulator **wchodzi za tablicę** i operuje
zza niej. Przyczyna w kodzie: geom `wall_panel` miał `contype = "0"` i
`conaffinity = "0"`, czyli był wyłącznie wizualny. Na sprzęcie omniboard jest
przykręcony do litej ściany i takie pozy są fizycznie niemożliwe — polityka
mogła więc uczyć się podejść, które **nie przenoszą się na robota**.

Zmiana: `contype/conaffinity = "1"`, a `_check_board_collision()` traktuje
`wall_panel` **identycznie jak płytę tablicy** (uderzenie w ścianę obok tablicy
nie jest bezpieczniejsze niż uderzenie w tablicę).

Sprawdzone, że włączenie kolizji nie generuje pasożytniczego kontaktu mimo
montażu na styk (tył tablicy i front ściany oba na `x = WALL_X = 0.50`):
`OmniBoardObject` ma `joints=None`, więc tablica i ściana są **przyspawane do
świata** i MuJoCo filtruje tę parę. Przyciski mają złącza przesuwne, ale leżą
na froncie (`x ≈ 0.42`), 8 cm od ściany. Test: 0 kontaktów ściana–tablica
i ściana–przycisk po resecie, `qacc` skończone, brak kontaktu w kroku 0.

Test blokowania: przy komendzie „w bok, potem prosto w ścianę" TCP dojeżdża do
`x = 0.492` przy ścianie zaczynającej się na `0.500` i dalej nie przechodzi;
51 kontaktów w 300 krokach, `_check_board_collision()` zwraca `True`.

**Konsekwencja dla porównań.** Modele z rozdz. 6.16–6.18 były uczone i mierzone
w środowisku **bez** ściany, więc ich liczby (80/100, 52/100) pochodzą z zadania
łatwiejszego niż obecne — część podejść zza tablicy była tam dozwolona. Wymiar
obserwacji się nie zmienia (ściana nie wchodzi do obs), więc stare modele nadal
się ładują, ale ich wyniki w nowym środowisku trzeba **zmierzyć od nowa**, zanim
zostaną z czymkolwiek zestawione.

**Pomiar kontrolny: ile ściana faktycznie zabrała.** `s2/best_model` przemierzony
w nowym środowisku, 100 sparowanych epizodów, protokół identyczny jak w 6.18
(`logs/board_6dof/s3/baseline_wall_100ep.csv`):

| metryka | bez ściany (6.18) | **ze ścianą** | zmiana |
|---|---|---|---|
| sukces | 52/100 | **51/100** | −1 pp (nieistotne) |
| kroki kolizji / epizod | 8.8 | **15.4** | **+75 %** |
| epizody z kolizją | 83/100 | 91/100 | +8 pp |
| misuse ramienia / epizod | 0.4 | 0.1 | −0.3 |

Wynik rozstrzyga interpretację: polityka **nie budowała skuteczności na
operowaniu zza tablicy** — sukces się nie ruszył. Wjeżdżała jednak w obszar, który
teraz zajmuje ściana, i te 6.6 dodatkowego kroku kontaktu na epizod było wcześniej
**niewidoczne i darmowe**. Luka sim2real była więc realna, choć mniejsza, niż
sugerowały nagrania. Nowy punkt odniesienia dla kolejnych biegów:
**51/100 przy 15.4 kroku kolizji/ep**, a nie 52/100 przy 8.8.

### 6.20. Bieg s3: obcięcie nagrody NIE zapobiega degradacji — wynik negatywny (2026-08-05)

Test hipotezy z rozdz. 6.19: jeśli kolaps z 6.18 brał się z konfliktu między
`depress_shaping_k` (nagradza głębszy docisk) a `collision_cost` (karze
wynikający z niego kontakt), to obcięcie nagrody na 13 mm powinno go usunąć.

Config `ppo_board_6dof_s3.yaml`, fine-tune z `s2/best_model.zip` (16.43M =
szczyt s2), budżet 3M. Wobec s2 zmieniły się **tylko dwie rzeczy, obie w env**:
obcięcie nagrody i fizyczna ściana. Reward v2.3 i hiperparametry PPO bez zmian.

**Przebieg — ten sam kształt co s2, tylko szybszy.**

| okno | sukces | pkt | reward eval |
|---|---|---|---|
| 16.4–16.9M | 51 % | 9 | −0.27 |
| 16.9–17.4M | **52 %** | 10 | −0.39 |
| 17.4–17.9M | 38 % | 10 | −0.55 |
| 17.9–18.4M | 27 % | 6 | −0.89 |

Bieg **przerwany na 18.21M** (1.78M z 3M) po spełnieniu kryterium stopu (b):
dwa kolejne pełne okna spadające o >8 pp (−14 pp, potem −11 pp). Kryterium
zadziałało tak, jak miało — w s2 taka sama degradacja kosztowała 3.5M kroków
zanim została zauważona.

**Ewaluacja 100 sparowanych epizodów** (`logs/board_6dof/s3/eval_100ep.csv`):

| model | sukces | kroki kolizji/ep | epizody z kolizją | misuse ep. |
|---|---|---|---|---|
| `s2_best` (punkt startowy) | 51/100 | **15.4** | 91/100 | 7/100 |
| `s3_ckpt17.33M` = `s3_best` | 52/100 | **20.4** | 95/100 | 15/100 |

Sukces: Fisher **p = 1**, pary rozbieżne 26:27 — modele **nierozróżnialne**.
Kolizyjność: **+5.0 kroku/epizod na niekorzyść s3** (54 epizody gorsze, 41
lepszych), misuse ramienia podwojony. `s3_best` znów tożsamy z checkpointem
szczytowym co do epizodu.

**Wniosek: hipoteza z 6.19 jest OBALONA.** Obcięcie nagrody działa dokładnie
tak, jak zaprojektowane (weryfikacja liczbowa w 6.19: 5.20 zamiast 10.00), ale
**nie wpływa na przebieg biegu** — degradacja wystąpiła i tak, wcześniej niż
w s2. Konflikt shaping-vs-kara nie był więc mechanizmem kolapsu, co najwyżej
jego towarzyszem.

**Obserwacja przeciwna do intuicji.** s3 uczył się **ze ścianą** i mimo to
koliduje z nią **częściej** (20.4) niż model, który ściany nigdy nie widział
(15.4). Kara `collision_cost = 5.0` nie przebija się więc przez gradient
zadania w tej konfiguracji — to nie jest kwestia „za mało kroków".

**Bilans trzech prób naprawy rewardem u 6DOF** (po trzech analogicznych
u 7DOF, rozdz. 6.5–6.6):

| próba | zmiana | wynik |
|---|---|---|
| s2 | `collision_cost` 0.5 → 5.0 | kolizyjność 4.7× w dół, ale kolaps po 2.5M |
| s3 | + obcięcie nagrody (13 mm) | bez zmian w zadaniu, kolizyjność **w górę** |

Sufit ~52 % przy pełnym v2.3 okazał się **odporny na obie interwencje**.
Zgodnie z linią z rozdz. 6.5–6.6 dalsze strojenie rewardu nie jest właściwą
drogą; kolejny krok powinien być **strukturalny** (horyzont, poza bazy,
parametry kontrolera) albo przenieść ciężar bezpieczeństwa do `SafetyLayer`
nadzorcy.

**Model na sprzęt pozostaje `logs/board_6dof/s2/best_model/best_model.zip`**
(51/100 ze ścianą, 15.4 kroku kolizji/ep). s3 nie dostarczył nic lepszego.

### 6.21. Sufit 52 % był w dużej mierze artefaktem horyzontu: 51 → 74/100 (2026-08-05)

Po trzech nieudanych interwencjach w reward (6.18–6.20) sprawdzono hipotezę
**strukturalną**, nie rewardową: czy limit 1000 kroków nie ucina zwycięstw.

**Przesłanka z danych** (`s2/best_model`, h=1000): rozkład czasu sukcesu był
**ucięty przy limicie** — mediana 705, 90. percentyl 899, maksimum **959**
kroków, czyli 41 kroków przed końcem horyzontu. Do tego 15 porażek miało
4/5 wciśniętych, a 13 kolejnych 3/5.

**Wynik** (ten sam model, 100 sparowanych epizodów, jedyna zmiana to horyzont):

| metryka | h = 1000 | **h = 1500** |
|---|---|---|
| sukces | 51/100 | **74/100** |
| średni czas sukcesu | 682 kr. (34 s) | 840 kr. (42 s) |
| kroki kolizji / epizod | 15.4 | 24.7 |
| porażki 4/5 → 3/5 → 2/5 | 15 → 13 → 10 | 8 → 7 → 6 |
| epizody `stuck` | 4 | 1 |

Fisher **p = 0.0012**, pary **23 : 0** — ani jeden epizod nie wypadł lepiej
przy krótszym horyzoncie (idealne zagnieżdżenie, jak dla tej samej polityki
z większym budżetem czasu). **23 sukcesy zajęły ponad 1000 kroków**, więc przy
starym protokole były niemożliwe.

> **SPROSTOWANIE (2026-08-06, rozdz. 6.23).** Zdanie zapisane tu pierwotnie —
> „ogon jest domknięty: maksimum 1429, h = 1500 wystarcza, h = 2000 nic by nie
> dodało" — jest **nieprawdziwe**. Pomiar przy h = 2000 dał **85/100**, w tym
> 11 sukcesów dłuższych niż 1500 kroków. Maksimum 1429 nie było końcem ogona,
> tylko ogonem **uciętym przez h = 1500** — czyli dokładnie ten sam błąd
> rozumowania, który ten rozdział wykrywa dla h = 1000, popełniony ponownie
> jeden akapit dalej.

**Wniosek.** Sufit ~52 %, z którym zmagały się rozdz. 6.18–6.20, był w znacznej
części **artefaktem protokołu pomiarowego**. Mierzył „ile robot zdąży w 50 s",
a nie „czy robot umie zadanie". Polityka z s2 umiała je w 74 % przypadków —
potrzebowała ~70 s zamiast 50 s. Nie unieważnia to wniosków o kolizyjności ani
o kolapsie (dotyczyły treningu, nie pomiaru), ale przesuwa ich wagę: znaczna
część wysiłku 6.18–6.20 goniła liczbę leżącą po stronie protokołu.

**Błąd w oszacowaniu z góry**: przewidywano
górną granicę +15 pp, licząc tylko epizody 4/5. Rozumowanie było wadliwe —
przy dodatkowych 500 krokach epizod z 3/5 może wcisnąć **dwa** przyciski.
Konwersja objęła cały rozkład porażek, nie tylko jego czoło.

**Zastrzeżenie metodologiczne.** Dla sprzętu wydłużenie horyzontu jest
akceptowalne (~70 s to realny czas zadania; nadzorca 7DOF z retry działał
75 s — rozdz. 5.1), ale w tabelach trzeba podawać horyzont przy **każdej**
liczbie sukcesu. Próba kontrolnego przemierzenia 7DOF przy obu horyzontach
ujawniła osobny, poważniejszy problem — rozdz. 6.22.

### 6.22. Baseline 7DOF (90/100) jest z INNEJ GEOMETRII i nie nadaje się do porównań (2026-08-05)

Żeby zestawienie 74/100 (6DOF, h=1500) z 90/100 (7DOF) było kontrolowane,
przemierzono model 7DOF `ppo_board_v23_12M.zip` na obecnym środowisku:

| model 7DOF v2.3 @12M | sukces | wciśnięcia per przycisk | misuse/ep |
|---|---|---|---|
| h = 1000 | **0/100** | 9 / 6 / 0 / 3 / 1 | 22.3 |
| h = 1500 | **0/100** | 10 / 6 / 1 / 3 / 1 | 31.4 |

Porażki wg wciśniętych: **83 epizody z zerem** przycisków. To nie jest regresja
modelu, tylko pomiar na **innym zadaniu**: model pochodzi z 19.07, a poprawki
geometrii z audytu (rozdz. 6.15) weszły 02.08 — zmieniły wysokość tablicy
(1.20 → 0.60 m), wysokość bazy (1.562 → 0.65 m) i dołożyły obrót 45°.
Do tego doszła fizyczna ściana (6.19). Polityka steruje ramieniem w miejsce,
gdzie tablica stała **przed** audytem.

**Konsekwencja — sprostowanie do rozdz. 6.17.** Zapisane tam zdanie, że różnica
80/100 (6DOF) wobec 90/100 (7DOF) to „realny koszt utraty stopnia swobody",
**nie ma pokrycia w danych**: obie liczby pochodzą z różnych wersji zadania.
Ta sama teza trafiła do aktualizacji wysłanej promotorowi i wymaga sprostowania.

**Kontrola i ablacja** (`training/diag_7dof_geometria.py`, ten sam model
w czterech wariantach geometrii, 25 epizodów każdy). Pytanie kontrolne było
zasadne: skoro obserwacja zawiera **względną** pozycję przycisku
(`target_relative_pos`) i **oś docisku** (`target_press_axis`), polityka
powinna w zasadzie przenieść się na nową geometrię.

| wariant | tablica | baza | roll | sukces | wciśnięć/ep |
|---|---|---|---|---|---|
| A — obecna | 0.60 | 0.65 | 45° | 0/25 | 0.16 |
| **B — stara (kontrola)** | 1.20 | 1.562 | 0° | **23/25 (92 %)** | 4.80 |
| C — tylko roll cofnięty | 0.60 | 0.65 | 0° | 0/25 | 0.12 |
| D — tylko wysokości cofnięte | 1.20 | 1.562 | 45° | 0/25 | **2.04** |

**B potwierdza, że model jest sprawny** (92 % ≈ historyczne 90/100), więc 0/100
w wariancie A to wyłącznie skutek zmiany zadania.

**Rozkład winy jest nierówny i pouczający:**

- **Obrót 45° polityka częściowo znosi** (D: 2.04 wciśnięcia/ep mimo obrotu,
  którego nigdy nie widziała). To zasługa względnej reprezentacji celu —
  `target_relative_pos` i `target_press_axis` niosą tę informację. Argument
  za tym, że projekt obserwacji pod sim2real był słuszny.
- **Zmiana wysokości jest zabójcza** (C: 0.12 wciśnięcia/ep). Obserwacja
  zawiera też **absolutne kąty stawów i pozycję EEF**; przeniesienie bazy
  o 0.91 m i tablicy o 0.60 m zmienia wysokość tablicy względem bazy
  z −0.36 m na −0.05 m. Wektor do celu wygląda znajomo, ale **postawa ramienia
  potrzebna do jego zrealizowania leży poza rozkładem treningowym**, a sieć
  (MLP na pełnym wektorze obs) nie przenosi wyuczonego odwzorowania do innego
  rejonu przestrzeni stawów.

Wniosek dla pracy: względna reprezentacja celu daje **odporność na zmianę pozy
tablicy**, ale **nie** na przeniesienie bazy robota. Przy sim2real oznacza to,
że wysokość montażu ramienia na Spocie musi się zgadzać z symulacją —
inaczej polityka nie przeniesie się nawet przy idealnej percepcji.

**Stan faktyczny:** dla obecnej geometrii **nie istnieje ważny baseline 7DOF**.
Żeby go uzyskać, trzeba by przetrenować 7DOF na poprawionej geometrii. Ablacja
wyżej pokazuje, że **scratch nie jest konieczny**: model działa (92 % w starej
geometrii), więc wystarczyłby fine-tune na nowej — rzędu 2–3M kroków (~5 h)
zamiast 10–12M od zera. Nadal jest to koszt trudny do uzasadnienia, skoro
robotem docelowym jest 6DOF. Rekomendacja: **nie odtwarzać baseline'u 7DOF**,
a w pracy opisać wyniki 6DOF jako samodzielne, z 7DOF przedstawionym jako
wcześniejszy etap metodyczny (na innej wersji zadania), nie jako punkt
odniesienia liczbowego.

**Lekcja.** Każda zmiana geometrii/fizyki env unieważnia WSZYSTKIE wcześniejsze
liczby, nie tylko te z bieżącego wątku. Modele ładują się bez błędu (obs się
zgadza), więc pomiar wygląda na poprawny — jedynym sygnałem jest absurdalnie
niski wynik. Przy każdej liczbie w pracy podawać **wersję env**, nie tylko
horyzont.

### 6.23. Horyzont 2000: 85/100 — sprostowanie do 6.21 i domknięcie wątku (2026-08-06)

Rozdz. 6.21 zamknął się tezą, że ogon rozkładu czasu sukcesu jest domknięty
przy h = 1500 (maksimum 1429 kroków) i h = 2000 nic by nie dodał. Teza była
**błędna**, i to w sposób pouczający: maksimum 1429 to nie koniec ogona, tylko
ogon ucięty przez sam mierzony horyzont. Ten sam model przemierzono więc przy
h = 2000 (`logs/board_6dof/s2/eval_h2000_100ep.csv`, log
`eval_horyzont_domkniecie.log`).

**Wynik** (`s2/best_model`, 100 sparowanych epizodów, jedyna zmiana to horyzont):

| metryka | h = 1000 | h = 1500 | **h = 2000** |
|---|---|---|---|
| sukces | 51/100 | 74/100 | **85/100** |
| średni czas sukcesu | 682 kr. (34 s) | 840 kr. (42 s) | 950 kr. (47.5 s) |
| mediana / 90. perc. / max | 705 / 899 / 959 | 799 / 1274 / 1429 | 891 / 1574 / **1926** |
| kroki kolizji / epizod | 15.4 | 24.7 | **32.7** |
| epizody z misuse | — | 9/100 | 9/100 |
| porażki 4/5 → 3/5 → 2/5 | 15 → 13 → 10 | 8 → 7 → 6 | 2 → 3 → 5 |
| epizody `stuck` | 4 | 1 | 1 |

Wciśnięcia per przycisk przy h = 2000 pozostają płaskie: 93 / 90 / 94 / 87 / 90.

**Istotność — dwa testy, dwa różne pytania.** Test Fishera na tabeli 85/15 vs
74/26 daje **p = 0.079**, czyli formalnie brak istotności. Ale Fisher zakłada
próby **niezależne**, a tu jest ta sama polityka na tych samych 100 seedach;
właściwy jest test parowany. Pary wypadają **11 : 0** (żaden epizod nie wypadł
lepiej przy krótszym horyzoncie), test dwumianowy dokładny **p = 0.00098**.
Uwaga praktyczna: `compare_models.py` raportuje Fishera, więc przy porównaniach
horyzontów jego p-wartość jest **zbyt konserwatywna** — trzeba czytać kolumnę
par, a nie samo p.

**Czy ogon jest teraz domknięty?** Ostrożniej niż poprzednio: maksimum 1926 to
74 kroki przed limitem, czyli rozkład **nadal dotyka granicy**. Przy h = 2500
doszłoby zapewne jeszcze kilka sukcesów. Różnica jest taka, że to już **nie
jest pytanie pomiarowe, tylko zadaniowe**: 2000 kroków to 100 s, a nadzorca
7DOF z retry pracował w 75 s (rozdz. 5.1). Dalsze wydłużanie mierzyłoby
cierpliwość, nie umiejętność.

**Kolizyjność rośnie proporcjonalnie do horyzontu** (15.4 → 24.7 → 32.7 kroku
na epizod, czyli ~1.6 kroku kolizji na 100 kroków epizodu niezależnie od
limitu). Dłuższy horyzont nie poprawia polityki — daje jej tylko więcej czasu,
w tym więcej czasu na ocieranie się o tablicę. Dla sprzętu jest to argument za
limitem prędkości/siły w `SafetyLayer`, nie przeciwko dłuższemu horyzontowi.

**Co raportować w pracy.** Rekomendacja: **74/100 przy h = 1500 (42 s)** jako
wynik główny, **85/100 przy h = 2000 (47.5 s śr., do 96 s w ogonie)** jako
pomiar uzupełniający pokazujący, gdzie leży sufit czasowy polityki. Uzasadnienie
wyboru h = 1500 jest zadaniowe, nie statystyczne: mieści się w budżecie czasu,
w którym nadzorca 7DOF osiągał 92/100 z retry.

**Lekcja metodologiczna (druga z tego samego wątku).** Wnioskowanie „maksimum
obserwowanego czasu leży poniżej limitu, więc limit nie wiąże" jest **błędne
przy rozkładach uciętych** — obserwuje się tylko epizody, które zdążyły, więc
maksimum zawsze wygląda na bezpieczne. Jedynym rozstrzygnięciem jest pomiar
przy dłuższym horyzoncie. Ten sam błąd popełniono tu dwa razy pod rząd
(h = 1000 → 1500 → 2000), za drugim razem **w akapicie wykrywającym go za
pierwszym**.

### 6.24. Webots jako źródło geometrii: zadanie przeniesione na wymiary areny + szum pod realną percepcję (2026-09-01)

Przed wyjściem na sprzęt zadanie postawiono w drugim symulatorze — `webots_ros2_spot`
(MASKOR / FH Aachen), gałąź `kinova_gen3`. Powód był początkowo wąski: napisać
percepcję (ArUco, później YOLO) i przetestować nadzorcę na Spocie z kamerami,
zanim dotknie się prawdziwego robota. Bieg okazał się mieć konsekwencje dla
samego środowiska treningowego.

**Co ta gałąź zawiera.** `Gen36dofWithRobotiq2f140.proto` + `GEN3-6DOF_VISION.urdf`
— czyli dokładnie robot uczelniany, 6DOF, nie 7DOF. Do tego gotowy
`webots_spot/detect_dex_board.py` (detekcja tablicy: HSV + głębia + broadcast TF),
`dex_sequence.py`, `nav2dex.py` i `servo_launch.py` (MoveIt Servo,
`servo_node/delta_twist_cmds`). Kamera nadgarstkowa Gen3 Vision publikuje jako
`Gen3/kinova_color` i `Gen3/kinova_depth` (640×480, `minRange` 0.001,
`maxRange` 10) — i to jest właściwa kamera dla tego zadania, bo realny Gen3 ma ją
fizycznie, więc transfer percepcji jest 1:1.

**Pomiar geometrii.** Wykonany na działającej symulacji przez TF, nie odczytany
z pliku świata:

```
/Spot/gps                      -> base_link          z = 0.5073
base_link -> gen3_base_link    = [0.256, 0.000, 0.112]   (bez obrotu)
=> baza ramienia                                     z = 0.619

gniazda przycisków (węzeł DEF estop, przeliczone przez łańcuch przekształceń):
  estop (środkowy)  z = 0.639      estop(1)  z = 0.714
  estop(4)          z = 0.716      estop(2)  z = 0.574
  estop(3)          z = 0.561
=> środek tablicy                                    z = 0.639
```

**Rozbieżność.** W robosuite tablica leżała 5 cm **poniżej** bazy ramienia
(`BOARD_Z` 0.60, `KINOVA_BASE_Z` 0.65). W Webots leży 2 cm **powyżej**. Różnica
7 cm — poza `board_pos_noise = 0.05`, więc nie do nadrobienia randomizacją.
Promień gniazd od środkowego: 0.106 / 0.110 / 0.110 (czwarte 0.092 odrzucone jako
niedokładność modelu) wobec przyjętych 0.120.

Obrót 45° **potwierdzony niezależnie**: kąty gniazd wyszły +45.0 / +135.4 /
−45.0 / −135.0 stopnia. Wysunięcie gniazda środkowego przed pozostałe: ~0.034
wobec naszych 0.040. Te dwie decyzje z audytu 6.15 były trafione.

**Decyzja: dostroić trening do Webots, nie odwrotnie.** Dotychczasowe 0.60 i 0.65
były **przyjęte, nie zmierzone** — 0.60 z widełek regulaminu Rescue League ("Omni
may be mounted on the wall at 60 cm or floor"), 0.65 z oszacowania "Spot ~0.5 m +
mocowanie ~0.15 m". Model MASKOR robi zespół mający fizyczną arenę z omniboardem,
więc jest bliżej celu niż nasze założenia. Hierarchia wiarygodności:
**realna arena > Webots > robosuite**, przy czym Webots też jest modelem — widać
to po tym czwartym gnieździe na 0.092, gdy trzy pozostałe leżą na 0.106–0.110.
Stąd zasada: brać z Webots **układ i rzędy wielkości, nie milimetry**; docelowo
zmierzyć tablicę miarką na uczelni.

**Zmiany w `envs/button_board.py`:**

| parametr | było | jest | podstawa |
|---|---|---|---|
| `BOARD_Z` | 0.60 | **0.64** | środek tablicy 0.639 |
| `KINOVA_BASE_Z` | 0.65 | **0.62** | baza ramienia 0.619 (TF) |
| promień gniazd | 0.120 | **0.107** | mediana z 0.106/0.110/0.110 |
| `BOARD_ROLL_DEG` | 45.0 | 45.0 | potwierdzone, bez zmian |

**Szum podniesiony pod realną percepcję.** Dotychczasowe poziomy opisywały odczyt
pozy z symulatora, nie z kamery. Na sprzęcie pozę tablicy poda ArUco:

| parametr | było | jest |
|---|---|---|
| `board_pos_noise` | 0.05 | 0.06 |
| `percep_pos_bias` | 0.005 | **0.015** |
| `percep_pos_noise` | 0.002 | **0.005** |
| `percep_axis_noise` | 0.02 | **0.05** |
| `base_z_noise` | — | **0.03** (nowy) |

Bias 15 mm to rząd błędu pozy znacznika ArUco na kamerze 640×480 z odległości
roboczej, 5 mm to drżenie między klatkami. `percep_axis_noise` podniesiony
**proporcjonalnie mocniej** niż pozycyjny, bo błąd orientacji z ArUco jest
znacznie gorszy od pozycyjnego: płaski znacznik ma niejednoznaczność pozy i
normalna potrafi przeskakiwać. Szum jest **wyłącznie w obserwacji** — nagroda i
detekcja wciśnięcia dalej liczą się na prawdzie z symulatora, inaczej polityka
uczyłaby się oszukiwać własny czujnik.

`base_z_noise` nie jest szumem percepcji, tylko odpowiedzią na **jedyny zmierzony
tryb awarii transferu** (6.22): względna reprezentacja celu chroni przed pozą
tablicy, ale nie przed przeniesieniem bazy, bo obs zawiera także bezwzględne kąty
stawów i pozycję EEF. Na Spocie wysokość korpusu jest sterowalna
(`/Spot/set_height`), a montaż nie będzie trafiony co do centymetra — więc ta oś
musi być w treningu, nie w kalibracji.

**Czego NIE zmieniono.** Geometrii samego grzybka: Webots ma Ø60 × 10 mm
wystający 75 mm nad gniazdo, nasz model Ø44 × 24 mm wystający 53 mm. Powód:
przyciski w Webots **nie mają fizyki** (`boundingObject` w całym `spot.wbt`
występuje 0 razy — kolidują wyłącznie obiekty `SolidBox`, których PROTO definiuje
bryłę wewnętrznie), więc ich wymiary są tak samo niezweryfikowane jak nasze. Za to
nasz model ma dostrojony skok 15 mm, `stiffness 500` i próg 12 mm, na których
wisi cała nagroda. Wymiana jednej niepotwierdzonej geometrii na drugą kosztowałaby
ponowne strojenie progu bez zysku.

**Weryfikacja nowej geometrii** (25 resetów, `Kinova3_6DOF`):

```
baza z:       0.591 .. 0.648   (randomizacja działa)
przyciski z:  0.478 .. 0.797
TCP startowy: x = 0.327,  z = 0.631 .. 0.688
KONTAKT W KROKU 0:  0/25 epizodów
```

Ostatnia liczba była głównym ryzykiem: `base_x_offset = -0.10` było strojone pod
starą geometrię (6.15) i mogło przestać chronić przed startem z penetracją
kapturków. Nie przestało.

**Konsekwencja pomiarowa.** Zmiana geometrii i szumu **unieważnia 74/100 przy
h = 1500** jako opis tego środowiska. Nie zestawiać wyników sprzed i po 01.09 —
to ten sam błąd, co porównywanie 7DOF 90/100 z 6DOF 80/100 (6.17, sprostowane
w 6.22). Model `s2/best_model` też nie jest ważną bazą na nowej geometrii; jeśli
potrzebny punkt odniesienia, zmierzyć go w NOWYM env przed startem fine-tunu i
podać jako "przed dostrojeniem".

**Czego nie zweryfikowano.** Czy wszystkie pięć przycisków jest osiągalne przy
najgorszej kombinacji randomizacji — przycisk na 0.797 przy bazie na 0.591 leży
21 cm powyżej bazy. Jeśli rozbicie per przycisk pokaże jeden ze stałym 0 %,
kandydatem jest zawężenie `base_z_noise` do 0.02.

**Baseline „przed dostrojeniem"** (`s2/best_model` w NOWYM env, h = 1500, 100
epizodów, seed 1234, `logs/board_6dof/s2/eval_geom_webots_h1500_100ep.csv`):

```
sukcesy:                     53/100        (na starej geometrii: 74/100 @h=1500)
sredni czas sukcesu:         892 krokow (44.6 s)
kolizje:                     100/100 ep., 37.6 kroku/ep   (bylo 24.7)
misuse przycisku:            10/100 ep., 0.3 kroku/ep     (bylo ~0)
gladkosc sr. ||da||/krok:    2.884
porazki wg wcisnietych 0..4: [4, 4, 11, 17, 11], stuck: 1
wcisniecia per przycisk:     btn0 80 | btn1 74 | btn2 82 | btn3 76 | btn4 74
```

Trzy wnioski, wszystkie istotne dla planu fine-tunu:

1. **Żaden przycisk nie wypadł.** Rozrzut 74–82/100 przy rozstępie 8 pp
   odpowiada na otwarte pytanie o osiągalność przy `base_z_noise = 0.03`:
   wszystkie pięć gniazd jest osiągalne także przy skrajnej randomizacji.
   Zawężanie `base_z_noise` do 0.02 jest **niepotrzebne**.

2. **To nie jest awaria typu „polityka poza rozkładem".** Z 47 porażek aż 39
   (83 %) miało wciśnięte 2 lub więcej przycisków, a 28 miało 3 lub 4. Tylko
   4 epizody skończyły się z zerem. Dla kontrastu: 7DOF po zmianie geometrii
   (6.22) dawał 0/100 przy **zerze** wciśnięć w 83 epizodach. Tutaj polityka
   wykonuje zadanie, tylko wolniej i mniej niezawodnie — to punkt wyjścia,
   z którego fine-tune ma z czego korzystać.

3. **Uwaga na horyzont — po raz trzeci.** 17 porażek z 4/5 wciśniętymi przy
   średnim czasie sukcesu 892 kroków to dokładnie sygnatura ucięcia
   horyzontem, opisana w 6.21 i 6.23. Zanim ta liczba wejdzie do pracy jako
   „53/100", trzeba ją przemierzyć przy h = 2000 — inaczej powtarza się błąd
   z 6.23, tam popełniony **w akapicie wykrywającym go za pierwszym razem**.

Wzrost kolizyjności 24.7 → 37.6 kroku/ep jest realny i istotny dla sprzętu:
tablica leży teraz wyżej względem bazy, więc ramię pracuje w innej postawie.
Do obserwacji w fine-tunie — `collision_cost = 5.0` był dobrany przy starej
geometrii.

**Domknięcie horyzontu dla nowej geometrii** (ten sam model, ten sam seed 1234,
100 sparowanych epizodów, `eval_geom_webots_h2000_100ep.csv`):

```
                        h = 1500        h = 2000
sukcesy                 53/100          66/100
sredni czas sukcesu     892 krokow      1063 krokow  (44.6 s / 53.1 s)
kolizje                 37.6 kroku/ep   50.7 kroku/ep
porazki wg wcisnietych  [4,4,11,17,11]  [4,2,9,12,7]
per przycisk            74..82          78..89

pary niezgodne: sukces TYLKO przy h=2000: 13 | TYLKO przy h=1500: 0
test dwumianowy (n=13, k=13, p=0.5):  p = 2.44e-04
```

Hipoteza z poprzedniego akapitu **potwierdzona**: 7 z 11 porażek, które przy
h = 1500 miały wciśnięte 4/5, zamieniło się w sukcesy przy h = 2000. Pary 13:0 —
żaden epizod nie stracił sukcesu przy dłuższym horyzoncie, co jest oczekiwane
przy horyzoncie jako jedynej zmiennej i potwierdza, że parowanie działa.

**Ogon NADAL jest ucięty.** Najdłuższy sukces przy h = 2000 trwał **1995 kroków
przy limicie 2000**, a 6 sukcesów wypadło w ostatnich 200 krokach. Dokładnie ta
sama sytuacja co w 6.23 (max 1429 przy limicie 1500). Czyli **66/100 też jest
dolnym oszacowaniem**, nie sufitem — h = 2500 najprawdopodobniej dołożyłby
kolejne. Reguła z [[lekcje-pomiarowe]] potwierdza się po raz trzeci: przy
rozkładach uciętych maksimum zawsze wygląda na bezpieczne, bo obserwuje się
tylko epizody, które zdążyły.

**Dwie różne liczby na dwa różne pytania** — nie mieszać ich w pracy:

- *„Czy polityka umie zadanie na tej geometrii?"* → **66/100 przy h = 2000**,
  i to wciąż dolne oszacowanie.
- *„Ile zdąży w budżecie nadzorcy?"* → **53/100 przy h = 1500**. 1500 kroków
  przy 20 Hz to dokładnie 75 s, czyli budżet nadzorcy z retry (rozdz. 5.1).
  h = 2000 to 100 s i wychodzi poza ten budżet.

Do oceny fine-tunu bazą jest **53/100 @ h = 1500**, bo to liczba operacyjna.
66/100 @ h = 2000 służy do stwierdzenia, że polityka nie zgubiła zadania po
zmianie geometrii — a nie zgubiła.

**Bieg**: `training/configs/ppo_board_6dof_webots_ft.yaml`, fine-tune 2M kroków
z `logs/board_6dof/s2/best_model/best_model.zip`, `learning_rate` 1e-4 (niższy
niż 3e-4 w s2 — chodzi o przesunięcie polityki, nie przebudowę), reward v2.3
nietknięty, żeby zmienną niezależną były wyłącznie geometria i szum. Budżet 2M
zamiast 6M, bo s2 kolapsował po ~2.5M, a s3 pokazał, że obcięcie nagrody tego nie
zmienia (6.20) — liczy się model ze szczytu, nie końcowy.

### 6.25. Eksplozja `log_std`: polityka bang-bang jako przyczyna plateau — 53 → 76/100 (2026-09-04…07)

Bieg `webots_ft` (fine-tune 2M na geometrii z 6.24) nie poprawiał wyniku, a jego
diagnostyka rosła monotonicznie: `approx_kl` 0.09 → 0.40, `clip_fraction` 0.67,
`std` 21.3 → 27.1 przy `mean_reward` bez trendu. To ta sama sygnatura co w 6.18
(s2 kolapsuje po ~2.5M) i 6.20 (s3 degraduje się szybciej). Zamiast po raz
czwarty zmieniać reward, zbadano **parametry samego rozkładu polityki**.

**Pomiar 1 — `log_std` rośnie z długością treningu, niezależnie od robota:**

```
model                          śr. std   max std
7DOF scratch_3M (wczesny)         1.00      1.01   <- wartość początkowa SB3
7DOF v4_6M                        5.89      7.29
7DOF v23_12M (90/100)            13.71     28.05
6DOF s2 (74/100 @h=1500)         21.32     48.90
```

**Pomiar 2 — akcje są nasycone** (`s2/best_model`, 200 kroków):

```
akcje stochastyczne (tak działa TRENING):     96.2 % składowych na granicy ±1
akcje deterministyczne (tak działa EWALUACJA): 95.9 %
```

**Pomiar 3 — surowa średnia sieci, przed obcięciem:**

```
zakres  -81.3 … +63.8      średnie |mu| = 14.80
udział |mu| <= 1 (w dopuszczalnym zakresie akcji):  4.1 %
```

**Wniosek: polityka jest przełącznikiem (bang-bang), nie regulatorem.** Sieć
wypluwa średnie kilkanaście razy poza `[-1, 1]`, obcięcie zamienia je w skrajne
wartości. Sterowanie proporcjonalne nie istnieje — ani w treningu, ani przy
ewaluacji deterministycznej.

**Mechanizm.** Entropia rozkładu Gaussa rośnie z `sigma` bez ograniczeń, więc
`ent_coef = 0.005` **stale pcha `log_std` w górę**. Równolegle SB3 obcina akcję
przed przekazaniem jej do środowiska, ale `log_prob` liczy na akcji **przed**
obcięciem — polityka może więc przesuwać średnią dowolnie daleko, bo zachowanie
i tak jest już nasycone i nic jej nie hamuje. Brak siły przywracającej.

**Dlaczego `ACTION_COST` tego nie złapał.** Kara `-ACTION_COST * sum(a^2)`
istnieje (0.01), ale jest **strukturalnie ślepa**: środowisko widzi wyłącznie
akcję po obcięciu, więc `sum(a^2)` nie przekroczy 7 niezależnie od tego, czy
średnia sieci wynosi 1 czy 80. Maksymalna kara to `0.07 / 225 = 0.0003` wobec
`PRESS_BONUS = 25 / 225 = 0.111` — **350x mniej**. Siła przywracająca musi
działać na politykę, nie na nagrodę.

**Co to wyjaśnia wstecz:**

- **dlaczego gracji nie dało się naprawić rewardem** (3 nieudane próby, 6.5–6.6,
  oraz `action_rate_cost` w v2.1): bang-bang **fizycznie nie potrafi** poruszać
  się gładko. Wniosek z 6.6 („gracji nie naprawiać rewardem, tylko limitem
  prędkości w SafetyLayer") był słuszny, ale z niewłaściwego powodu — problem
  nie leżał w funkcji celu, tylko w reprezentacji polityki;
- **dlaczego fine-tune degraduje**: parametry rozkładu dryfują bez konsekwencji
  behawioralnych, stąd `approx_kl` 0.40 przy płaskim wyniku;
- **dlaczego plateau jest odporne na strojenie rewardu** (6.18–6.20): zmienia się
  funkcję celu, a polityka i tak dysponuje wyłącznie przełącznikiem.

**Czego to NIE wyjaśnia — kontrola.** `v23_12M` osiągnął 90/100 przy `std = 13.7`,
więc eksplozja `std` **nie uniemożliwia sukcesu**. Nie jest to zatem jedyna
przyczyna plateau, a hipoteza „to wyłącznie wina 6DOF" pozostaje nierozstrzygnięta.
Ustalone jest natomiast, że reward nie był właściwym miejscem naprawy, a różnica
`std` 13.7 vs 21.3 idzie w parze z różnicą wyników.

**Odpowiedź na pytanie „czy SAC byłby lepszy".** Dla tej konkretnej patologii —
tak: SAC z konstrukcji używa gaussa ściśniętego przez `tanh`, więc średnia nie
może uciec poza zakres akcji, i stroi entropię do zadanego celu zamiast dodawać
premię rosnącą bez ograniczeń. Nie jest to jednak powód do przepisania projektu:
PPO tu działa (53–66/100), a przyczynę można usunąć w samym PPO.

**Interwencja.** Dodano `LogStdClamp` (`training/train_board.py`) — callback
ograniczający `log_std` po każdym rolloucie, sterowany parametrem `log_std_max`
w YAML. Ograniczenie `sigma` ogranicza też pośrednio ucieczkę średniej: przy
małym `sigma` próbki nie odbiegają daleko, więc `mu` nie ma za czym gonić.
Dodano też obsługę `target_kl` (wcześniej nieobsługiwaną) oraz `eval_freq` i
`n_eval_episodes` z configu.

#### Wyjaśnienie mechanizmu (dla rozdziału metodycznego pracy)

**Polityka nie zwraca akcji, tylko rozkład.** Sieć daje dwie wielkości: `mu`
(średnia — „w co celuję") i `sigma` (rozrzut — „jak bardzo losuję wokół"). W
treningu akcja jest losowana z `N(mu, sigma)` dla eksploracji; w ewaluacji brane
jest samo `mu`. Zakres akcji to `[-1, 1]`, wszystko poza nim jest obcinane.

**Wpływ `sigma` na to, co faktycznie robi robot** (polityka celuje w `mu = 0.3`,
czyli łagodny ruch; 200 tys. próbek):

| `sigma` | % próbek w zakresie | przykład 8 wylosowanych akcji |
|---|---|---|
| 0.2 | 100.0 % | `0.31 0.37 0.13 0.29 0.05 0.10 0.42 0.27` |
| 0.5 | 91.4 % | `-0.05 -0.19 1.00 0.85 -0.20 0.71 0.87 0.07` |
| 1.0 | 66.0 % | `1.00 -0.33 -1.00 1.00 0.87 -0.73 0.84 -1.00` |
| 5.0 | 15.8 % | `-1.00 -1.00 1.00 1.00 1.00 -1.00 0.20 -0.64` |
| **21.3** | **3.6 %** | `-1.00 -1.00 -1.00 -1.00 1.00 1.00 1.00 -1.00` |

Przy `sigma = 0.2` akcje krążą wokół zamierzonych 0.3 — polityka robi to, co
zamierza. Przy `sigma = 21.3` (wartość zmierzona w `s2/best_model`) próbki są
rozrzucone w zakresie ok. ±60, więc po obcięciu **każda ląduje na −1 albo +1**;
zamiar `0.3` przestaje mieć znaczenie. Wiersz 3.6 % zgadza się z niezależnym
pomiarem na modelu (4.1 % surowych średnich w zakresie).

**Bang-bang** to sterowanie wyłącznie wartościami skrajnymi, bez stanów
pośrednich — odpowiednik kierowcy dysponującego tylko pełnym gazem i pełnym
hamulcem. Taki kierowca dojedzie do celu, szarpiąc i przejeżdżając, ale nie
zaparkuje precyzyjnie. To jest dokładnie profil trudności zadania: dolot do
przycisku i docisk na 12 mm wymagają sterowania proporcjonalnego.

**Dlaczego `sigma` rosło — to jest reward hacking, tylko nie w nagrodzie
zadania, lecz w regularyzatorze PPO.** Entropia rozkładu Gaussa to w
przybliżeniu `log(sigma) + const`, więc premia `ent_coef = 0.005` rośnie wraz z
`sigma`. Kluczowe jest to, że **gdy `sigma` przekroczy ok. 2, dalszy wzrost nie
zmienia już zachowania** (akcje i tak są obcinane do ±1), ale nadal zwiększa
premię entropijną. Polityka znalazła darmowy dochód: podnoś `sigma`, inkasuj
premię, nie płać niczym. Ta sama asymetria dotyczy średniej — `mu = 5` i
`mu = 50` dają identyczne zachowanie po obcięciu, więc nic nie ciągnie `mu` z
powrotem do zakresu i uciekła do `|mu| = 14.8`.

**Skutek dla interpretacji całego projektu:** kolejne zmiany funkcji nagrody
(6.5–6.6, 6.18–6.20) modyfikowały **cel**, podczas gdy polityka nie dysponowała
**środkami** do jego realizacji. Przy `sigma = 21` polityka przestaje być
funkcją stanu — wykonuje w przybliżeniu to samo losowanie niezależnie od tego,
co obserwuje.

**Co daje ograniczenie `log_std <= 0` (czyli `sigma <= 1`).** Zamyka darmowy
dochód: próbki wracają w okolicę `mu`, więc wybór `mu` znowu przekłada się na
zachowanie — a to jest warunek konieczny sterowania proporcjonalnego. Pośrednio
ogranicza też ucieczkę średniej, bo przy małym `sigma` próbki nie odbiegają
daleko i gradient nie ma za czym gonić `mu`.

**Dlaczego to prawdopodobnie nie wystarczy przy fine-tunie.** Ściągnięcie
`sigma` z 21 do 1 nie zmienia faktu, że `mu` jest już przy 14.8 — akcje
pozostaną nasycone, tylko przestaną być losowe. Fine-tune może być więc
nienaprawialny i rozstrzygnięcia dostarczą dopiero biegi od zera (`sc1`, `sc2`),
w których polityka startuje z `sigma = 1` i `mu` w zakresie.

**Przemiat weryfikujący** (`training/run_sweep.sh`, `training/configs/sweep/`),
6 biegów, reward v2.3 i geometria nietknięte:

| bieg | typ | zmiana |
|---|---|---|
| `f1_targetkl` | fine-tune 1.5M | `target_kl 0.03` — kontrola, `std` nietknięte |
| `f2_kl_ent` | fine-tune 1.5M | `target_kl 0.03` + `ent_coef 0.001` |
| `f3_clamp` | fine-tune 1.5M | `ent_coef 0` + `log_std <= 0` (`std <= 1.0`) |
| `f4_clamp_tight` | fine-tune 1.5M | `ent_coef 0` + `log_std <= -0.7` (`std <= 0.5`) |
| `sc1_ent0` | **scratch 3M** | `ent_coef 0` + `log_std <= 0` |
| `sc2_ent0_kl` | **scratch 3M** | `ent_coef 0` + `log_std <= 0` + `target_kl 0.03` |

Scratche są tu istotne: wszystkie fine-tune'y startują z `s2/best_model`, czyli
**dziedziczą** `std = 21.3` i `|mu| = 14.8`. Obniżenie `ent_coef` może ściągnąć
`sigma`, ale średniej prawdopodobnie nie przywróci — skoro akcje są nasycone, nie
ma gradientu, który by ją cofnął. Fine-tune może być nienaprawialny i wtedy
odpowiedzi udzielą dopiero biegi od zera.

Wyniki lądują w `logs/board_6dof/sweep/WYNIKI.md` (sukces @h=1500 na 100
epizodach plus końcowe `std` każdego modelu). Baza do pobicia: **53/100**.

#### Wyniki przemiatu (2026-09-05) — hipoteza NIE potwierdzona w tej formie

Sześć biegów zakończonych, ewaluacja 100 sparowanych epizodów przy h = 1500,
ten sam seed 1234 co baza. Do każdego modelu domierzono `sigma`, surową średnią
polityki `|mu|` i udział akcji nasyconych:

| bieg | `sigma` | `\|mu\|` | % `mu` w `[-1,1]` | % akcji na ±1 | sukces | p vs baza |
|---|---|---|---|---|---|---|
| `s2` (baza) | 21.32 | 14.80 | 4.1 % | 95.9 % | **53/100** | — |
| `f3_targetkl` | 21.92 | 15.80 | 4.1 % | 96.0 % | 57/100 | 0.63 |
| `f1_clamp` | 1.00 | 14.61 | 5.2 % | 94.8 % | 50/100 | 0.75 |
| `f2_clamp_tight` | 0.50 | — | — | — | 44/100 | 0.20 |
| `sc1_loose` | 1.00 | 0.88 | 65.6 % | 35.1 % | 15/100 | 1.4e-08 |
| `ft_sc1_v23` | 1.00 | 1.06 | 52.6 % | 48.0 % | 17/100 | 2.1e-07 |
| `sc2_full` | 0.48 | 0.52 | 86.9 % | 13.2 % | 0/100 | 3.2e-13 |

**Wynik negatywny nr 1: żaden fine-tune z `s2` nie różni się istotnie od bazy.**
`f3` = 57 (p = 0.63), `f1` = 50 (p = 0.75), `f2` = 44 (p = 0.20); pary `f3` vs
`f1` dają p = 0.36. Sześć wariantów hiperparametrów, zero ruchu. Strojenie
hiperparametrów PPO na tej polityce **nie działa** — tak samo jak wcześniej nie
działało strojenie nagrody.

**Wynik negatywny nr 2: teza „bang-bang jest przyczyną plateau" upada w tej
formie.** Zależność jest wręcz odwrotna i monotoniczna — im bardziej
proporcjonalna polityka, tym gorszy wynik. `sc2_full` ma najlepsze sterowanie ze
wszystkich (86.9 % `mu` w zakresie, tylko 13.2 % akcji nasyconych) i **0/100**.

**Interwencja zadziałała mechanicznie.** `LogStdClamp` zrobił dokładnie to, co
miał: w biegach od zera `sigma` została na 1.00, `|mu|` na 0.88–1.06, a nasycenie
spadło z 96 % do 35–48 %. Sterowanie proporcjonalne zostało przywrócone.
Przewidywanie z poprzedniego akapitu też się potwierdziło: w `f1_clamp` `sigma`
spadła do 1.00, ale `|mu| = 14.61` zostało i akcje pozostały nasycone w 94.8 % —
**`f1` i `f2` w ogóle nie testowały hipotezy**, tylko potwierdziły, że fine-tune
zdegenerowanej polityki jest nienaprawialny.

**`sc2_full` = 0/100 potwierdza konieczność kurikulum.** `sigma` spadła sama do
0.48, czyli *poniżej* clampa — polityka z własnej woli ograniczyła eksplorację
i zamarła. To przewidywany kolaps „nie ruszaj się" przy `collision_cost = 5.0`
i rzadkiej nagrodzie od kroku 0, zgodny z historią (`s1_contaminated`,
`s1_clean`: 3–6 %).

#### Konfundent, przez który wynik NIE jest rozstrzygnięty

Nasycenie jest **idealnie skorelowane z długością treningu**, bo eksplozja
`sigma` rozwija się przez miliony kroków:

```
s2, f1, f3     16.4M kroków   nasycone 95-96 %      53-57/100
sc1             4.0M kroków   proporcjonalne 35 %       15/100
ft_sc1_v23      5.9M kroków   proporcjonalne 48 %       17/100
```

Nie da się rozdzielić „nasycenie pomaga" od „więcej kroków pomaga". Co więcej,
zestawienie z historyczną krzywą uczenia (6.17) sugeruje, że biegi z clampem są
**szybsze, nie gorsze**:

| budżet | historia (bez clampa) | z clampem |
|---|---|---|
| 4.0M | — | `sc1` **15 %** |
| 6.0–6.5M | 0 % | `ft_sc1_v23` (5.9M) **17 %** |
| 6.5–7.0M | 3 % | |
| 7.0–7.5M | 9 % | |
| 8.0–8.5M | 16 % | |

Historyczna ścieżka potrzebowała ~8M kroków na 16 %; `sc1` osiągnął 15 % przy
4M, a `ft_sc1_v23` 17 % przy 5.9M.

#### Test rozstrzygający: hipoteza POTWIERDZONA przy dopasowanym budżecie (2026-09-07)

`training/configs/sweep/ft_sc1_v23_long.yaml` — wznowienie `ft_sc1_v23` z 5.9M
o kolejne 10M, do **15.91M**, czyli budżetu `s2/best_model` (16.4M). Kryterium
postawione przed biegiem: dojście w okolice 53/100 obala hipotezę, wyraźne
przebicie ją potwierdza.

**Przebieg — przełamanie nastąpiło dopiero po 8M kroków:**

| odcinek | `ep_len_mean` | zmierzony sukces @h=1500 |
|---|---|---|
| 5.9–7.5M | 984–990 | 17/100 (przy 5.9M) |
| 7.5–8.0M | 983 | |
| 8.0–8.5M | 973 | |
| 8.5–9.0M | 962 | |
| 9.0–10.0M | 920 | 26/100 (przy 9.5M) |
| 10–11M | 874 | |
| 11–12M | 836 | |
| 12–13M | 774 | |
| 13–14M | 728 | |
| 14–15.9M | 721 | **70/100** (model końcowy) |

Nachylenie `ep_len_mean` od 7.5M: −37.6 kroku/1M przy `r = −0.85`; od 9.0M już
−42.2 przy `r = −0.92`. Pierwsze 1.7M biegu było płaskie (nachylenie **+7.1**,
`r = +0.40`) i ocena wykonana wtedy — przy 7.59M — wypadła jednoznacznie
negatywnie. Był to ostatni płaski punkt **tuż przed przełamaniem**.

**Wynik końcowy** (`ppo_board_final.zip`, 15.91M kroków, 100 sparowanych
epizodów, seed 1234, ten sam co baza):

| metryka | `s2` (16.4M) | clamp (15.91M) | zmiana |
|---|---|---|---|
| sukces @h=1500 | 53/100 | **70/100** | +17 pp, pary 37:20, **p = 0.033** |
| sukces @h=2000 | 66/100 | 77/100 | +11 pp, pary 25:14, p = 0.108 (n.s.) |
| średni czas sukcesu @h=1500 | 892 kroków (44.6 s) | **495 kroków (24.8 s)** | 1.8x szybciej |
| epizody z kolizją | 100/100 | **43/100** | — |
| kroki kolizji / epizod | 37.6 | **6.6** | 5.7x mniej |
| gładkość `\|\|da\|\|`/krok | 2.884 | **1.229** | 2.3x gładziej |
| misuse przycisku | 10/100 | 5/100 | — |
| per przycisk | 74–82 | 79–90 | lepiej i równiej |
| `stuck` | 1 | 6 | **jedyna regresja** |

Poprawa przy h = 1500 jest istotna (p = 0.033). Przy h = 2000 kierunek ten sam,
ale **nieistotny** (p = 0.108) — przy dłuższym horyzoncie baza nadrabia część
strat, więc różnica maleje. Podawać obie liczby.

**Ogon po raz pierwszy w projekcie NIE jest ucięty.** Przy h = 2000 najdłuższy
sukces trwał 1901 kroków przy limicie 2000, powyżej 1500 kroków wpadło 7
sukcesów, a w ostatnich 200 krokach tylko **1**. We wszystkich wcześniejszych
pomiarach maksimum lądowało tuż pod limitem (1429/1500, 1995/2000) i oznaczało
ogon ucięty (6.21, 6.23, 6.24). Tutaj **77/100 jest wynikiem realnym, nie dolnym
oszacowaniem** — co jest spójne ze średnim czasem sukcesu 495 kroków.

**Liczba operacyjna do pracy: 70/100 @ h = 1500** (1500 kroków = 75 s = budżet
nadzorcy z retry). Średni sukces w 24.8 s zostawia nadzorcy duży zapas na retry;
przy bazie było to 44.6 s.

#### Sprostowanie do wniosku z przemiatu

Wniosek zapisany 2026-09-05 („teza *bang-bang jest przyczyną plateau* upada")
był **przedwczesny i wymaga sprostowania**. Konfundent — nasycenie idealnie
skorelowane z długością treningu — został wtedy poprawnie zidentyfikowany
i opisany, ale mimo to sformułowano wniosek tak, jakby hipoteza była obalona.
Przemiat porównywał modele po 4–6M krokach z modelem po 16.4M; przy dopasowanym
budżecie kierunek jest odwrotny do tego, co z niego wyczytano.

Ten sam błąd powtórzył się w skali jednego biegu: ocena przy 7.59M („brak
trendu, `ep_len` rośnie, prawdopodobnie ślepa uliczka") padła 0.5M kroków przed
przełamaniem. **Reguła: przy rzadkiej nagrodzie i shapingu progowym płaski
odcinek nie jest dowodem sufitu.** Kryterium stopu musi być określone
z góry w krokach, a nie ustalane w reakcji na bieżący wykres — to ta sama
dyscyplina, którą wymusza reguła o podawaniu horyzontu.

Trzecia lekcja, tym razem narzędziowa: `eval/mean_reward` **prowadzi na
manowce**, gdy rośnie kara kolizyjna wraz z aktywnością polityki — w tym biegu
spadał (−0.44 na ostatnim odcinku), podczas gdy sukces rósł 26 → 70/100.
`best_model` wybierany po tym wskaźniku pochodził sprzed przełamania. Wszystkie
liczby wyżej pochodzą z **modelu końcowego**, nie z `best_model`. To rozszerzenie
obserwacji z 6.16 na przypadek, w którym reward eval jest zdominowany przez kary,
a nie przez shaping.

#### Wnioski

1. **Eksplozja `log_std` była realną przyczyną, a nie ciekawostką.** Jej
   usunięcie dało +17 pp skuteczności, 5.7x mniej kroków kolizji i 1.8x krótsze
   epizody — przy tym samym rewardzie, tej samej geometrii i porównywalnym
   budżecie. Jedyna zmiana to jeden callback ograniczający `log_std`.
2. **Wątek gracji jest domknięty.** Trzy próby naprawy nagrodą zawiodły
   (v2.2 → 0/100, v2.4 → zapaść 85→43 %, v2.5 → 83/100), a wniosek z 6.6 brzmiał
   „nie naprawiać rewardem, tylko limitem prędkości w SafetyLayer". Był słuszny
   co do rewardu, ale z niewłaściwego powodu: problem leżał w **reprezentacji
   polityki**. Przy `|mu| = 14.8` i 96 % nasyconych akcji gładki ruch był
   fizycznie niewykonalny. Clamp dał 2.3x gładszy ruch bez dotykania nagrody.
3. **Kurikulum pozostaje konieczne** — `sc2_full` (pełne v2.3 od kroku 0
   z clampem) dał 0/100 przy `sigma` opadłej samoczynnie do 0.48. Działająca
   ścieżka to `scratch na luźnych warunkach` → `zacieśnianie fine-tunem`.
4. **Clamp kosztuje tempo, nie sufit.** Biegi z clampem uczą się wolniej na
   początku (17/100 przy 5.9M, 26/100 przy 9.5M), ale nie zatrzymują się tam,
   gdzie polityka nasycona.
5. **Nowy model na sprzęt**:
   `logs/board_6dof/sweep/ft_sc1_v23_long/ppo_board_final.zip` — lepszy
   jednocześnie w skuteczności, szybkości i bezpieczeństwie kontaktowym.
   Zastępuje `s2/best_model` w tej roli.

#### Kontynuacja do 19M: sufit potwierdzony, degradacja zachowania (2026-09-07)

`ft_sc1_v23_long2` — wznowienie modelu 15.91M o kolejne 3M, do **18.93M**
(`target_kl` uciął ostatnią aktualizację, tak jak w poprzednim biegu).

| metryka | 15.91M | 18.93M | test |
|---|---|---|---|
| sukces @h=1500 | **70/100** | 62/100 | pary 14:22, p = 0.243 (n.s.) |
| sukces @h=2000 | **77/100** | 66/100 | |
| `stuck` | **6** | 17 | pary 13:2, **p = 0.0074** |
| epizody z kolizją | **43/100** | 54/100 | |
| kroki kolizji / epizod | **6.6** | 7.2 | |
| gładkość | 1.229 | 1.181 | |
| per przycisk | **79–90** | 72–82 | |

Sam spadek skuteczności **nie jest istotny** (p = 0.243) i nie należy go raportować
jako pogorszenie wyniku. Istotna jest natomiast **regresja zachowania**: `stuck`
wzrósł niemal trzykrotnie (p = 0.0074) — model znacznie częściej wpada w stany
bez wyjścia. Towarzyszy temu wzrost kolizyjności i pogorszenie na wszystkich
pięciu przyciskach.

Krzywa zapowiadała to wcześniej: nachylenie `ep_len_mean` spadło z −42.2 na
**−10.2** kroku/1M, a ostatni odcinek (18.5–19.0M) **zawrócił w górę** (679 po
632 na odcinku 18.0–18.5M).

**Wniosek: sufit tej konfiguracji leży w okolicach 70/100 @h=1500 i jest
osiągany około 16M kroków.** Dalszy trening nie dokłada skuteczności, a psuje
zachowanie — ten sam wzorzec co kolaps `s2` po ~2.5M (6.18), tylko łagodniejszy:
clamp spowalnia degradację, ale jej nie usuwa.

**Model na sprzęt pozostaje ten z 15.91M**
(`logs/board_6dof/sweep/ft_sc1_v23_long/ppo_board_final.zip`). Model z 18.93M
zachowany jako materiał dowodowy granicy.

Pozostaje otwarte: `stuck` 1 → 6 przy samym przejściu na politykę proporcjonalną
(niezależnie od degradacji przy 19M). Hipoteza: polityka nasycona „przepycha się"
przez stany bez wyjścia siłą, proporcjonalna nie ma takiej możliwości. Do
obejrzenia przez `watch_board.py --record` na epizodach oznaczonych `stuck`
w `eval_final_h1500.csv`.

#### Sprostowanie pomiarowe: zaszyty `board_pos_noise` w narzędziach (2026-09-07)

Przy diagnozowaniu epizodów `stuck` wyszło, że skrypt odtwarzający nie
reprodukował epizodów z CSV. Przyczyna: **`compare_models.py` i `watch_board.py`
miały zaszyte `board_pos_noise=0.05`**, nieaktualizowane przy podniesieniu szumu
z 01.09 (6.24, `0.05 → 0.06`). Pomiar szedł więc w warunkach *innych* niż
trening. Ta sama klasa błędu co w 6.22 — parametr zduplikowany w dwóch miejscach
i zmieniony tylko w jednym, bez żadnego objawu poza cichą rozbieżnością.

Naprawa: oba narzędzia **nie podają już tych parametrów wcale**, biorą domyślne
z `ButtonBoardEnv`. Jedno źródło prawdy; kolejna zmiana szumu automatycznie
obejmie pomiar.

**Przemiar obu modeli w warunkach treningowych** (100 sparowanych epizodów,
seed 1234, h = 1500, identyczne ustawienia dla obu):

| metryka | `s2/best_model` (16.4M) | `ft_sc1_v23_long` (15.9M) | zmiana |
|---|---|---|---|
| sukces @h=1500 | 53/100 | **76/100** | pary 34:11, **p = 8.2e-04** |
| średni czas sukcesu | 891 kroków (44.5 s) | **565 kroków (28.3 s)** | 1.6x szybciej |
| epizody z kolizją | 97/100 | **44/100** | — |
| kroki kolizji / epizod | 37.2 | **7.9** | 4.7x mniej |
| gładkość `\|\|da\|\|`/krok | 2.936 | **1.236** | 2.4x gładziej |
| misuse przycisku | 19/100 | **5/100** | 3.8x mniej |
| per przycisk | 69–76 | **83–89** | — |
| `stuck` | 1 | 9 | jedyna regresja |

Poprzednie liczby (53 vs 70) pochodziły z pomiaru przy `board_pos_noise=0.05`.
Wniosek się nie zmienił — **usterka nie zafałszowała porównania**, bo dotyczyła
obu modeli jednakowo; zmienił się zestaw wylosowanych epizodów. Różnica 70 vs 76
mieści się w rozrzucie próby (przy `n = 100` i `p ≈ 0.75` odchylenie to ok.
4.3 pp). **Obowiązujące liczby to te z tabeli wyżej.**

Uwaga metodyczna: `compare_models.py` raportuje test **Fishera** (tu p = 1.08e-03),
który zakłada niezależność prób. Epizody są sparowane seedem, więc właściwy jest
test dwumianowy na parach (p = 8.24e-04). Tutaj oba prowadzą do tego samego
wniosku, ale w 6.23 rozbieżność była dramatyczna (Fisher 0.079 wobec dwumianowego
0.00098). **Do rozważenia: dodać test na parach do samego narzędzia.**

#### Nagrania (2026-09-07)

- `docs/video/6dof_clamp_15.9M_76pct.mp4` — model końcowy, 6 epizodów,
  seed 1234 (parowanie z `eval_h1500_pelny_szum.csv`);
- `docs/video/6dof_stuck_*.mp4` — epizody `stuck` do diagnostyki.

Kadr kamery `board_view` jest o ~4 cm za nisko: dostrojony pod `BOARD_Z = 0.60`,
a tablica stoi na 0.64 od 01.09. Wpływa wyłącznie na nagrania, nie na obserwacje
ani nagrodę.

#### Diagnoza `stuck`: nadgarstek na limicie stawu `j4` (2026-09-07)

Punktem wyjścia była obserwacja z nagrań: ramię zawisa nad przyciskiem i
**obraca nadgarstkiem w lewo-prawo**, nie zbliżając się do celu. Oscylacja przy
nieruchomym TCP jest sygnaturą utraty stopnia swobody, nie błędu sterowania.

**Pomiar** (`ppo_board_final.zip`, 9 epizodów `stuck` vs 10 udanych,
odtworzonych po seedzie i zweryfikowanych wobec nagrań — liczba wciśniętych
przycisków zgadza się co do epizodu):

| grupa | % czasu z `j4` w 3° od limitu |
|---|---|
| `stuck` (n = 9) | **73.6 %** |
| sukces (n = 10) | 20.0 % |
| Mann-Whitney (stuck > sukces) | **p = 0.0031** |

Osiem z dziewięciu utknięć spędza 80–92 % epizodu z `j4` opartym o limit
(`[-2.10, +2.10]` rad, margines 0.00–0.04 rad). Stawy `j0`, `j3`, `j5` są
**ciągłe** (`jnt_range` = 0, czyli bez ograniczenia), ograniczone są tylko
`j1`, `j2`, `j4`.

Struktura ruchu w zablokowanym stanie (amplitudy w ostatnich 300 krokach):

```
              ograniczone            ciagle
ep17     j1=0°  j2=0°  j4=1°    j0=1°  j3=4°  j5=36°
ep45     j1=0°  j2=0°  j4=1°    j0=1°  j3=4°  j5=22°
```

`j5` (obrót nadgarstka) jest **jedynym stawem, który się jeszcze porusza** —
i to jest obserwowane „kręcenie nadgarstkiem". Polityka próbuje kompensować
jedynym wolnym stopniem swobody nadgarstka, ale zablokowana oś to `j4`, więc
ruch `j5` nie przybliża do celu.

**To nie jest problem orientacji, tylko zasięgu.** W dwóch z trzech
przeanalizowanych epizodów `eef_align_with_target` wynosi 8–12° (czyli
orientacja jest dobra), a TCP stoi 7.7–8.6 cm od celu. Ramię wie, gdzie jest
przycisk, i jest poprawnie ustawione — po prostu nie może się dalej wysunąć.

**Hipoteza „polityka celuje w widmo" — OBALONA.** Sprawdzono per-epizodowy bias
percepcji (`percep_pos_bias`, do ±15 mm):

| grupa | `\|bias_pos\|` |
|---|---|
| `stuck` | 14.4 mm |
| sukces | **15.6 mm** (większy!) |
| Mann-Whitney | p = 0.48 |

Sukcesy mają średnio *większy* błąd percepcji niż utknięcia. Polityka jest na
bias odporna — przyczyna jest wyłącznie kinematyczna.

**Blokada jest zależna od ścieżki, nie od geometrii.** Dwa z trzech
przeanalizowanych utknięć dotyczyły `btn0` — przycisku środkowego, czołowego,
geometrycznie najłatwiejszego i trafianego w 85/100 epizodów. Skoro to samo
gniazdo jest zwykle osiągalne, blokada wynika z konfiguracji, w jaką ramię
wpadło po drodze, a nie z nieosiągalności celu. Wspiera to obserwacja, że
w udanych epizodach `j4` też dotyka limitu (20 % czasu), ale ramię z niego
wychodzi.

**Konsekwencja dla nadzorcy:** istnieje **mierzalny wyzwalacz retry**, dużo
lepszy niż ogólne „brak postępu" — `j4` w granicach ok. 3° od limitu przez N
kolejnych kroków. `joint_states` Kinovy podaje `j4` wprost, więc detektor działa
identycznie w symulacji i na sprzęcie. Szacowany pułap: 9 z 24 porażek to
utknięcia, więc nawet idealny retry daje **~85/100**, nie 100/100.

Zastrzeżenie sim2real: `percep_pos_bias` jest w symulacji **stały na epizod**,
więc retry wewnątrz epizodu go nie zmienia. Na sprzęcie ArUco przeliczy pozę po
odjeździe, ale tylko jeśli zmieni się **punkt obserwacji** — retry z tej samej
pozy kamery odtworzy ten sam błąd. Retry musi więc obejmować zmianę punktu
widzenia, nie samo cofnięcie i powtórzenie ruchu.
### 6.26. Sim2sim Webots — warstwa A: kontroler. Zmierzona rozbieżność mechanizmów (2026-09-10)

Pierwszy etap przenoszenia polityki do Webots: **czy kontrolery mówią tym samym
językiem**. Bez tego reszta transferu nie ma sensu — akcje polityki znaczyłyby
w obu symulatorach co innego.

**Wybrana ścieżka.** Gen3 w Webots jest sterowany przez
`kinova_joint_trajectory_controller`, czyli w przestrzeni **stawów**. Polityka
wypuszcza deltę **kartezjańską** (OSC_POSE). Tłumaczenie realizuje usługa MoveIta
`/compute_ik`: poza TCP + delta → poza docelowa → kąty stawów → `JointTrajectory`.

Odrzucono MoveIt Servo (brak pakietu `ros-humble-moveit-servo`, dodatkowa czarna
skrzynka) oraz sterowanie w przestrzeni stawów à la `rl_bridge/gen3_rl_jointPos`
— tamten węzeł dodaje akcje wprost do kątów stawów, co jest poprawne **tylko dla
polityk uczonych w `JOINT_POSITION`**. Nasza jest uczona w OSC_POSE; dodanie
metrów do radianów dałoby ruch w losowym kierunku.

Narzędzie: `ros2_bridge/diag_ik_move.py`.

**Weryfikacja łańcucha.** `/compute_ik` odpowiada `error_code.val = 1` dla pozy
bieżącej oraz dla przesunięć ±5 cm w każdej osi, zwracając za każdym razem
istotnie różne kąty (|Δkątów| 0.32–0.77 rad). KDL wystarcza, `pick_ik`
niepotrzebny. Ramka komend: `gen3_base_link` (= `input_ref_frame: "base"` z OSC).

**Pomiar — pojedynczy ruch +0.05 m w Z, `time_from_start = 0.05 s`:**

| t [s] | Δz [m] | v [m/s] | % celu |
|---|---|---|---|
| 0.069 | 0.0074 | 0.116 | 15 % |
| 0.135 | 0.0287 | **0.323** | 57 % |
| 0.196 | 0.0394 | 0.175 | 79 % |
| 0.262 | 0.0447 | 0.080 | 89 % |
| 0.400 | 0.0487 | 0.019 | 97 % |
| 0.722 | 0.0500 | 0.002 | 100 % |

Zażądano 1.000 m/s (5 cm w 50 ms), osiągnięto 0.323 m/s w szczycie, pełne
przemieszczenie po **0.72 s**. Profil: rozpędzenie i asymptotyczne dojście do
celu.

**Przyczyna — limity modelu, nie kod.** `protos/Gen36dofWithRobotiq2f140.proto`
ogranicza silniki: `maxVelocity 1.3963` rad/s dla `joint_1..3` i `1.2218` dla
`joint_4..6`, momenty 39 / 9 Nm. Do tego `moveit_joint_limits.yaml` ma
`max_acceleration: 0.7` rad/s², co samo dałoby 1.34 s na tę zmianę konfiguracji
(0.316 rad). To jest **realistyczne** — prawdziwy Gen3 też ma takie limity,
a robosuite ich nie modeluje.

**Kluczowy wniosek: to są dwa jakościowo różne mechanizmy sterowania.**

| | robosuite OSC_POSE | Webots IK + JointTrajectoryController |
|---|---|---|
| natura | prędkość ciągła (regulator impedancyjny) | dojazd do zadanej pozycji |
| przy stałej komendzie | jedzie bez końca, 0.263 m/s | dojeżdża 5 cm i **staje** |
| w jednym kroku polityki (50 ms) | 0.0132 m | ok. 0.004 m |

W robosuite akcja `1.0` w Z znaczy „jedź w górę z prędkością 0.263 m/s i nie
przestawaj". W Webots ta sama komenda znaczy „przesuń się o 5 cm i zatrzymaj".
Średnia prędkość całego ruchu wychodzi **3.8x niższa** niż ustalona prędkość
w robosuite.

Konsekwencja dla transferu: polityka wypuszcza nową akcję co 50 ms, więc będzie
gonić cel, którego nie dogania, a `target_relative_pos` będzie maleć wolniej,
niż nauczyła się oczekiwać. To jest **pierwsza zmierzona luka sim2sim** i musi
zostać skompensowana albo świadomie przyjęta.

**Zastrzeżenie pomiarowe okazało się kluczowe.** Powyższe to pojedynczy dojazd.
W pętli sterowania każda nowa komenda przerywa poprzedni dojazd w fazie
rozpędzania, więc ramię nigdy nie wchodzi w wyhamowywanie — i wynik jest
zupełnie inny.

#### Pomiar w pętli — wniosek odwrócony (`ros2_bridge/diag_ik_loop.py`)

Nowa poza docelowa liczona co krok z bieżącego TCP, publikacja przy 20 Hz,
`set_ready_position` między osiami (bez resetu drugi pomiar startuje już poza
zasięgiem i daje same `IK FAIL`).

| oś | Webots IK+JTC | robosuite OSC | stosunek | 90 % @krok | sprzężenie |
|---|---|---|---|---|---|
| X | −0.01140 | −0.01126 | **1.01x** | 5 | 4.6 % |
| Y | +0.01065 | +0.00722 | **1.48x** | 1 | 1.9 % |
| Z | +0.01206 | +0.01317 | **0.92x** | 2 | 2.0 % |

**Kontrolery są praktycznie równoważne w pętli.** Pojedynczy dojazd sugerował
3.8x wolniej; w pętli wszystkie osie mieszczą się w 0.92–1.48x. `output_max =
0.05` można przenieść **bez przeliczania** — akcja polityki daje w Webots ruch
tego samego rzędu.

**Różnice między osiami się jednak odwróciły.** W robosuite Y było
najwolniejsze (0.00722 — OSC pokonuje większą masę zredukowaną w bok),
w Webots jest najszybsze (0.01065). To potwierdza mechanizm: **OSC czuje
bezwładność, IK liczy czystą kinematykę.** Stąd 1.48x na Y i 0.92x na Z.

Sprzężenie na pozostałe osie spadło z 13.4 % (Y, robosuite) do 1.9 % — IK
realizuje żądany kierunek dokładniej, bo rozwiązuje pozycję, a nie generuje siłę.

**Ocena ryzyka dla transferu: niska.** Rozbieżność rzędu ±30 % w tempie ruchu
jest mniejsza niż zaburzenia, na które polityka jest już odporna
(`board_pos_noise` 0.06, `percep_pos_bias` 0.015, `base_z_noise` 0.03).

**Dwa realne ryzyka, oba zidentyfikowane:**

1. **Granica zasięgu jest cicha.** Oś X wyczerpała zasięg po 11 krokach, Z po
   25. Zanim IK zwróci `-31`, oddaje **rozwiązania częściowe**: przyrost w osi
   zadanej spada (0.0124 → 0.0079 m/krok), a sprzężenie boczne rośnie
   40-krotnie (0.0002 → 0.0080 m/krok). Ramię jedzie w bok zamiast tam, gdzie
   kazano, **bez żadnego sygnału błędu**. Polityka tego nie wykryje.
2. **Reset między próbami jest konieczny** — w pętli sterowania musi go
   zapewnić nadzorca (`ros2 run webots_spot set_ready_position`).

**Napotkane pułapki ROS** (do rozdziału o implementacji):
- **discovery DDS** — publisher utworzony i użyty w ciągu kilkunastu ms nie ma
  jeszcze subskrybenta; wiadomość przepada bez śladu. Trzeba odczekać, spinując;
- **`time.sleep()` nie przetwarza wiadomości** — węzeł „widzi" świat wyłącznie
  gdy spinuje. Sen blokuje wątek, bufor TF zamarza, pomiar czyta stan sprzed
  ruchu i pokazuje zerową deltę;
- **bufor TF nie jest gotowy po starcie** — `TransformListener` zapełnia się
  dopiero podczas spinowania; pierwsze `lookup_transform` zawsze rzuca wyjątkiem;
- **`GetPositionIK.Request` ma jedno pole `ik_request`** — wszystkie parametry
  są zagnieżdżone. Struktury wiadomości sprawdzać przez
  `ros2 interface show <typ>`, nie zgadywać;
- odpowiedź IK zawiera **28 stawów** (palce chwytaka + napędy nóg Spota) —
  filtrować po nazwie, nigdy po indeksie;
- `rclpy` logger **nie przyjmuje obiektu wyjątku**, tylko `str`; przekazanie
  wyjątku rzuca `TypeError` wewnątrz bloku obsługi błędu i maskuje przyczynę.


#### Warstwa B — rozbieżność ramek EEF i modelu chwytaka (2026-09-10)

Przy budowie wektora obserwacji z ROS wykonano **test kinematyki prostej**: te
same kąty stawów wstawiono do obu symulatorów i porównano wynikową pozycję
chwytaka. Kąty odczytano z Webotsa (`/joint_states`), wstawiono do
`sim.data.qpos` w MuJoCo, wywołano `sim.forward()`.

```
katy                    [0.00, -1.90, -2.54, 0.00, -1.00, 1.57]
robosuite  eef - baza   [ 0.1114,  0.0014,  0.3828]
Webots  TF base->eef    [-0.0330,  0.0010,  0.3930]
roznica                 [ 0.1444,  0.0004, -0.0102]   |d| = 0.1448 m
```

Y i Z zgadzają się (0.4 mm i 10 mm), X różni się o 14.4 cm. **To nie jest błąd
kinematyki** — kolejność stawów, znaki i zera są poprawne. Oś Z ramki
`end_effector_link` wskazuje w tej pozie wzdłuż `[0.998, 0, −0.069]`, a
`0.1448 m` w tym kierunku daje `[0.1445, 0, −0.0100]`, czyli dokładnie zmierzoną
rozbieżność. Przesunięcie jest **wzdłuż osi narzędzia**.

**Ramki EEF znaczą co innego w obu symulatorach:**

| | ramka | odległość od kołnierza |
|---|---|---|
| robosuite | `gripper0_right_grip_site` — punkt chwytu między palcami | 0.1448 m |
| Webots | `end_effector_link` — kołnierz narzędzia | 0.000 m |
| Webots | `*_inner_finger_pad` — wkładki palców | 0.167 m |

Wstawienie `end_effector_link` wprost jako `eef_pos` przesunęłoby cały
`target_relative_pos` o 14 cm — przy `percep_pos_bias = 0.015` to katastrofa.

**Osobna rozbieżność: model chwytaka nie odpowiada laboratorium.**

```
robosuite      Robotiq 85     punkt chwytu 0.1448 m      <- zgodne z laboratorium
laboratorium   Robotiq 85
Webots         Robotiq 2f_140 wkladki palcow 0.167 m     <- NIEzgodne
```

Różnica 2.2 cm jest **fizyczna**, nie do usunięcia wyborem ramki. Co istotne,
zainstalowany pakiet `ros-humble-robotiq-description` dostarcza **2f_85**
(`robotiq_2f_85_gripper.urdf.xacro` + komplet meshy); katalog `2f_140` powstał
z ręcznej kopii wykonanej wg punktu 3 README MASKOR (6.24). `resource/gen3.urdf`
odwołuje się do `2f_140` w liniach 246–269.

**Decyzja:** kompensować w moście — `eef_pos` liczyć jako
`end_effector_link + 0.1448 m` wzdłuż jego lokalnej osi Z, co odtwarza punkt
chwytu Robotiq85, zgodny **i z treningiem, i z realnym robotem**. Podmiana
modelu chwytaka w Webots na 2f_85 zostaje jako zadanie **przed pomiarami
kontaktowymi** (fizyka przycisków), bo tam geometria wkładek zacznie mieć
znaczenie. Dla warstwy B, która sprawdza sam transfer polityki, długość chwytaka
w Webots jest bez wpływu, o ile obserwacja jest poprawna.

**Uwaga metodyczna:** rozbieżność wykryto porównaniem kinematyki prostej, a nie
lekturą dokumentacji. To jest tania kontrola (kilkanaście minut), którą warto
wykonać przy każdym przenoszeniu polityki między symulatorami — sama zgodność
liczby stawów i nazw ramek niczego nie gwarantuje.

**Druga rozbieżność, wykryta przy okazji:** `/joint_states` ma w Webots **dwóch
wydawców publikujących rozłączne zestawy stawów** — `spot_driver` (16 napędów
nóg) i `joint_state_broadcaster` (6 stawów Gen3 + `finger_joint`). Zapamiętywanie
„ostatniej wiadomości" daje kod, który losowo działa i losowo rzuca
`ValueError`. Trzeba **akumulować wartości po nazwach**. Do tego kolejność
w wiadomości jest przemieszana (`joint_1, joint_2, joint_4, joint_5, joint_3,
joint_6`) — filtrowanie po indeksie zamieniłoby `joint_3` z `joint_4` bez
żadnego objawu.

**Chwytak w obserwacji:** robosuite oczekuje `gripper_qpos[6]` i
`gripper_qvel[6]` (kinematyka palców Robotiq85), a Webots wystawia **jeden**
staw (`finger_joint`). Odczyt z Webotsa jest więc niemożliwy. Przy
`force_gripper_closed=True` chwytak był zamknięty przez cały trening, więc
w moście wstawiane są mediany z robosuite (300 kroków, chwytak zamknięty):
`qpos [0.4117, −0.0919, 0.0036, 0.6345, 0.2445, 0.2810]`,
`qvel [−2.05, −22.43, −1.39, −0.06, −10.07, −0.07]`. Fluktuacja jest duża
(`qvel` sięga 113 w szczytach), mediana stabilna.

**`eef_quat_site`** nie ma odpowiednika w Webots — to ramka wewnętrzna
robosuite. Zweryfikowano na 60 pozach, że jest to `eef_quat` złożony ze
**stałym** obrotem −90° wokół lokalnej osi Z (kwaternion `[0, 0, −0.7071,
0.7071]`, odchylenie macierzy 7.5e-07). Liczona w moście, nie odczytywana.

#### B0 domknięte: most obserwacji odtwarza wektor treningowy (2026-09-11)

`ros2_bridge/obs.py` buduje 120-wymiarowy wektor obserwacji z ROS i porównuje go
z robosuite **przy tych samych kątach stawów** (odczytanych z Webotsa, wstawionych
do `sim.data.qpos`, `sim.forward()`). Szum percepcji w env wyzerowany.

| blok | max \|różnica\| |
|---|---|
| `joint_pos` / `cos` / `sin` / `vel` | 0.000000 |
| `eef_pos` | 0.000175 |
| `eef_quat` | 0.000003 |
| `eef_quat_site` | 0.000003 |
| `target_relative_pos` | 0.000175 |
| `target_distance` | 0.000130 |
| `target_onehot` | 0.000000 |
| `target_press_axis` | 0.000000 |
| `eef_align` | 0.000009 |
| `gripper_qpos` / `qvel` | **rozbieżne — patrz niżej** |

Dwanaście z czternastu bloków zgadza się co do **0.2 mm**. Pozostałe dwa nie mogą
się zgodzić z przyczyn sprzętowych.

**Dwie usterki wykryte dopiero tym porównaniem** — żadna z wcześniejszych kontroli
(długość wektora, duplikat, brak `NaN`, zgodność kinematyki prostej) ich nie łapała:

**1. `eef_pos` był w złej ramce.** W robosuite `obs[24:27]` jest **bezwzględne**
w układzie świata, nie względem bazy ramienia. Most zwracał pozycję w
`gen3_base_link`, czyli przesuniętą o `[-0.10, 0, 0.62]`. Naprawa: stała
`BASE_IN_WORLD`. To nie jest kosmetyka — bezwzględna poza EEF jest dokładnie tym
mechanizmem, który zabił transfer 7DOF po przeniesieniu bazy (6.22).
`target_relative_pos` był odporny, bo różnica wektorów skraca początek układu.

**2. Znak kwaternionu.** `q` i `−q` opisują ten sam obrót, więc wszystkie testy
kątowe pokazywały 0°. Ale **sieć widzi liczby**. Pomiar wpływu na `ppo_board_final`
(150 kroków, odwrócony znak `eef_quat` i `eef_quat_site`):

```
srednia |roznica akcji| : 0.5004     (zakres akcji [-1, 1])
maks   |roznica akcji|  : 2.0000     (pelne odwrocenie)
% krokow z roznica >0.1 : 100.0 %
```

Konwencje odtworzono z kodu robosuite i zweryfikowano na 200 pozach:

| obserwacja | źródło w robosuite | reguła znaku |
|---|---|---|
| `eef_quat_site` | `T.mat2quat(site_xmat)` | `w >= 0` (mat2quat wymusza jawnie) |
| `eef_quat` | surowy `body_xquat` z MuJoCo | największa co do modułu składowa **ujemna** |

Przy okazji ujawniła się znana niespójność robosuite (issue #298, opisana
w docstringu `robots/robot.py`): **`eef_pos` czytane jest z `site`, a `eef_quat`
z `body`** — dwie różne rzeczy w jednym wektorze obserwacji. `eef_quat_site`
dodano później jako spójną wersję, ale `eef_quat` zachowano dla zgodności ze
starymi zbiorami danych. Most musi odtwarzać **obie**.

**Świadome uproszczenie: `gripper_qpos` / `qvel`.** Robosuite oczekuje po 6
wartości (kinematyka palców Robotiq85), Webots wystawia **jeden** staw
(`finger_joint`) — odczyt jest niemożliwy. Przy `force_gripper_closed=True`
chwytak był zamknięty przez cały trening, więc wstawiane są mediany z robosuite.
Fluktuacja jest duża (`qvel` sięga 113 w szczytach), mediana stabilna.

> **Sprostowanie (rozdz. 6.27).** Wpisane wówczas wartości `qvel` nie były
> medianą, tylko **odchyleniem standardowym ze znakiem minus**. Błąd przeszedł
> walidację B0, bo ta porównywała pozycje bloków, a nie odpowiedź polityki.
> Wykryty dopiero w B1 przez porównanie akcji. Poprawione wartości i ocena
> wrażliwości polityki na ten blok — rozdz. 6.27.

**Metodycznie:** kontrola „wektor ma poprawną długość, duplikat się zgadza, nie
ma `NaN`" jest **niewystarczająca**. Obie usterki przechodziły ją bez zarzutu.
Dopiero porównanie **element po elemencie z referencyjnym symulatorem, przy tej
samej konfiguracji stawów**, je ujawniło. To jest kontrola warta wykonania przy
każdym przenoszeniu polityki — koszt kilkunastu minut, a wykrywa klasę błędów,
która inaczej objawia się dopiero jako niewyjaśniona utrata skuteczności.

### 6.27. Sim2sim Webots — warstwa B1: polityka podłączona, obserwacja zwalidowana akcją (2026-09-11)

Warstwa B0 (rozdz. 6.26) domknęła **budowanie** wektora obserwacji z ROS-a.
B1 dokłada politykę, ale **nie wysyła akcji** — most liczy `model.predict()`
i wypisuje wynik. Celem jest odpowiedź na pytanie, którego B0 nie mogło
postawić: czy polityka, karmiona tym wektorem, zachowuje się tak samo jak
w robosuite.

**Narzędzie:** `ros2_bridge/policy_bridge.py`, tryb `--dry-run`.

#### Bazowanie: poza startowa jest częścią kontraktu

Pierwsze uruchomienie dało dystans do celu **0.508 m**, po ręcznym podniesieniu
ramienia **0.892 m**. Pomiar w robosuite (8 ziaren, pełne epizody):

| wielkość | wartość |
|---|---|
| dystans na resecie | 0.182 ± 0.026 m |
| **największy dystans w całym epizodzie** | **0.296 m** |
| nasycenie akcji na resecie | 68 % |
| nasycenie średnie w epizodzie | 48 % |

Polityka **nigdy w treningu nie widziała dystansu większego niż 0.30 m**.
Most podawał jej 0.5–0.9 m, czyli ekstrapolację.

Przyczyną było to, że Webots startuje z ramieniem złożonym (TCP w ramce bazy
`[-0.0332, 0.0014, 0.3928]`), a robosuite resetuje do pozy wyciągniętej
(`[0.4265, 0.0014, 0.0396]`). Różnica 0.46 m w osi `x`.

Poza resetu w `ButtonBoardEnv` jest **stała** — odchylenie standardowe zero na
wszystkich sześciu stawach przez 8 ziaren:

```
joint_pos [rad] : [0.0, 0.65, -1.89, 0.0, -0.60, -1.5708]
joint_pos [deg] : [0, 37.2, -108.3, 0, -34.4, -90.0]
```

**Wniosek metodyczny:** poza startowa jest elementem kontraktu sim2real na
równi z układem wektora obserwacji. Polityka trenowana z jednego resetu jest
zdefiniowana tylko w otoczeniu trajektorii z tego resetu. Most dostał metodę
`home()`, która wysyła te sześć kątów wprost na `JointTrajectory`, **z pominięciem
`/compute_ik`**: nie ma wtedy dwuznaczności konfiguracji łokcia, orientacja
wychodzi sama, a `obs[0:6]` zgadza się co do bitu. `time_from_start` dla
bazowania to 3 s, nie 0.05 s jak w pętli sterowania.

Efekt uboczny wart odnotowania: `home()` weryfikuje przy okazji **kolejność
`JOINTS`**. Rozbieżność na jednym stawie to za krótki czas przejazdu albo
ogranicznik; rozbieżność na dwóch naraz to przestawiona kolejność nazw.

#### Poza robocza wygląda nienaturalnie i jest poprawna

Po zbazowaniu ramię w Webots układa się z przedramieniem w górę i chwytakiem
skierowanym **pionowo w dół**. Wygląda to na błąd, a jest wyuczonym zachowaniem:

```
oś Z narzędzia w świecie : [0.002, 0.000, -1.000]
eef_quat                 : [-1, 0, 0, 0]
eef_align_with_target    : 1.5692 rad = 89.9°
```

`eef_align` bliskie 90° znaczy, że chwytak jest **prostopadły** do osi
wciskania. Polityka naciska **kostkami palców przy chwytaku skierowanym w dół**,
a nie osią narzędzia w przycisk. Jest to spójne z rewardem v2.3, który premiuje
kontakt palców i karze kontakt ogniw ramienia (rozdz. 6.18).

#### Usterka: stałe chwytaka to było odchylenie standardowe, nie mediana

Po zbazowaniu dystans i poza zgadzały się z robosuite, ale akcja nie:

| | robosuite | most |
|---|---|---|
| `a[0:3]` | `[-1.0, -0.578, -1.0]` | `[0.898, -0.251, -0.599]` |
| `\|a\|` średnie | 0.940 | 0.450 |
| nasycenie | 86 % | 0 % |

Test podmiany, przeprowadzony **w robosuite** przez wstawienie stałych mostu
do referencyjnego wektora obserwacji, wskazał sprawcę jednoznacznie:

| podmieniony blok | `a[0:3]` | `\|a\|` |
|---|---|---|
| nic (oryginał) | `[-1.0, -0.578, -1.0]` | 0.940 |
| `gripper_qpos` | `[-1.0, -0.357, -1.0]` | 0.791 |
| **`gripper_qvel`** | `[0.912, -0.272, -0.740]` | **0.458** |
| oba | `[0.899, -0.251, -0.598]` | **0.450** |

Ostatni wiersz odtwarza wynik mostu co do trzeciego miejsca po przecinku.
Oznacza to, że **cały pozostały wektor — 108 z 120 wymiarów, w tym oba
kwaterniony, oś wciskania i kąt wyrównania — był poprawny od początku**.

Przyczyna: `GRIPPER_QVEL_MED` zawierało **odchylenie standardowe ze znakiem
minus**, nie medianę.

| | wpisane | zmierzone (6 epizodów, 2977 kroków, szum treningowy) |
|---|---|---|
| `GRIPPER_QPOS_MED` | `[0.4117, -0.0919, 0.0036, 0.6345, 0.2445, 0.2810]` | `[0.4301, -0.2016, 0.2189, 0.6243, 0.1544, 0.2708]` |
| `GRIPPER_QVEL_MED` | `[-2.0529, -22.4271, -1.3936, -0.0558, -10.0700, -0.0734]` | `[-0.0009, -0.0103, 0.0017, 0.0011, -0.0050, 0.0001]` |

Zmierzone `std` dla `qvel` to `[2.48, 24.99, 1.68, 0.14, 11.36, 0.14]` — co do
wartości bezwzględnej pokrywa się z błędnymi stałymi. Chwytak przy
`force_gripper_closed=True` **drga wokół zera** z dużym rozrzutem; mediana jest
bliska zeru, a nie kilkunastu rad/s.

**To jest samodzielne ryzyko sim2real.** Blok chwytaka to 12 z 60 unikalnych
wymiarów, a sama podmiana prędkości zmienia `|a|` z 0.940 na 0.458. Polityka
reaguje na niego **nieproporcjonalnie mocno** w stosunku do jego rozmiaru, a most
podaje go jako **stałą**, bo Webots ma 2f_140 o innej kinematyce niż Robotiq85
z treningu. Na sprzęcie z Robotiqiem 85 należy odczytać wartości rzeczywiste.

#### Wynik B1

Po poprawce stałych, poza `home`, cel btn0, `--dry-run`:

| | wzorzec robosuite | most w Webots |
|---|---|---|
| `a[0:3]` | `[-1.0, -0.217, -1.0]` | `[-1.0, -0.216, -1.0]` |
| `\|a\|` średnie | 0.786 | 0.785 |
| nasycenie | 71 % | 71 % |
| dystans | 0.045 | 0.045 |
| kroki z `NaN` | — | 0 |

**Warstwa B1 domknięta.** Ścieżka ROS → obserwacja → polityka odtwarza
robosuite.

#### Fałszywy alarm wart odnotowania

W trakcie diagnozy `TRAINING_POS` (pozycje gniazd w ramce bazy) zostało błędnie
uznane za niezgodne z env. Pomiar porównawczy wykonano na pierwszym site
z `important_sites` obiektu przycisku, podczas gdy `_setup_observables` liczy
`target_relative_pos` na site **`btn{i}_button_top`**, czyli na czubku kapturka.
Oba site różnią się o kilka centymetrów w osi `x`. Po pomiarze na właściwym
site wartości w moście okazały się poprawne co do czwartego miejsca po przecinku.

Wniosek: przy walidacji krzyżowej między symulatorami trzeba porównywać
**dokładnie ten punkt, którego używa obserwacja**, a nie dowolny site o
zbliżonej nazwie. Zapis `envs/obs_contract.json` podaje znaczenie pola, ale nie
nazwę site — to luka w kontrakcie.

#### Znane odstępstwo do domknięcia w B2

Most podaje **geometrię treningową**, nie rzeczywistą płytę z Webotsa:
`button_pose()` zwraca stałe z robosuite. Na B1 to jest właściwe, bo izoluje
ścieżkę obserwacji od percepcji. Po zamknięciu pętli ramię pojedzie w miejsce
z robosuite, a nie tam, gdzie stoi omniboard w symulacji — objawi się to jako
mijanie przycisku o kilka centymetrów i będzie wyglądać na błąd polityki.
Do B2 potrzebne jest przeliczenie współrzędnych świata Webotsa do ramki bazy,
czyli poza Spota.

Kontrola dostępna od razu: różnica btn0→btn1 zmierzona w świecie Webotsa to
`(-0.039, +0.075, +0.075)`, a pół-rozstaw gniazd w env wynosi `0.0757`
(= promień `0.107` / √2). Kształt płyty w obu symulatorach się zgadza —
nieznana pozostaje wyłącznie poza bazy względem niej.

### 6.28. Warstwa B2 — orientacja jest obowiązkowa, a mechanizmy się nie pokrywają (2026-09-11)

**Sprostowanie do 6.26 i do zaleceń z B1.** Zapisana wcześniej konkluzja
warstwy A („skale osi 0.92–1.48× robosuite, `output_max = 0.05` przenosi się
bez przeliczania") **nie jest wiarygodna**. Pomiar wykonano ścieżką, która nie
odejmowała `GRIP_OFFSET`, czyli celowała kołnierzem 14.5 cm za daleko. Po
poprawieniu tej ścieżki zmierzona skala translacji przy `time_from_start=0.05`
wynosi **0.55×**, nie 0.92×.

**Orientacji nie da się odłożyć.** Zalecenie z B1, żeby domknąć najpierw
pozycję, jest błędne dla tej polityki. Wyzerowanie `a[3:6]` w robosuite
odtwarza awarię obserwowaną w Webots: dystans ucieka `0.045 → 0.331`, a
`eef_align` stoi w okolicy 85°. Z pełną akcją `align` schodzi z 89.9° do ~24°
w 30 krokach przy prawie stałym dystansie. **Strategią tej polityki jest obrót
nadgarstka, nie dojazd pozycją.**

Złożenie delty obrotu zaimplementowano na kwaternionach i zweryfikowano
przeciwko `robosuite.utils.transform_utils` na 500 losowych przypadkach
(maks. błąd 2.3e-07). Kolejność to `R_cel = R_delta @ R_bieżące`, czyli
mnożenie z lewej, zgodnie z `input_ref_frame = "base"`.

**Drugie sprostowanie:** przypuszczenie, że czysty obrót powinien zostawiać
punkt chwytu nieruchomo, jest błędne. OSC w robosuite też go przesuwa, i to
bardziej (0.342 m wobec 0.171 m w moście przez 25 kroków).

**Kalibracja mechanizmu.** Jedynym nastawnikiem jest `time_from_start`:

| `tfs` [s] | transl. `+Z` | × robosuite | obrót `+X` | × robosuite |
|---|---|---|---|---|
| 0.02 | 0.00992 | 0.77 | 4.10° | **1.01** |
| 0.03 | 0.00962 | 0.74 | 2.87° | 0.71 |
| 0.05 | 0.00718 | 0.55 | 2.76° | 0.68 |
| 0.08 | 0.00455 | 0.35 | 2.40° | 0.59 |

Przyjęto `tfs = 0.02` i `POS_SCALE = 0.065` zamiast 0.05, co daje translację
1.05× robosuite. Kierunki obrotu zweryfikowano osobno — oś Z narzędzia idzie
w tę samą stronę dla `rot X`, `rot Y` i `rot Z`.

**Wzmocnienie obrotu per oś.** Zapisane wyżej „`+X` = 1.01×" jest **błędne** —
mierzono przechyleniem osi Z narzędzia, a ta miara zaniża obrót wokół samej osi
narzędzia. Poprawną miarą jest kąt geodezyjny pełnej orientacji,
`arccos((tr(Rᵃᵀ Rᵇ) − 1) / 2)`. Pomiar 20 kroków od pozy home, `tfs = 0.02`:

| oś | robosuite [°/krok] | most [°/krok] | stosunek | błędy IK |
|---|---|---|---|---|
| `+X` | 3.946 | 3.416 | 0.87 | 0 |
| `−X` | 3.975 | 3.570 | 0.90 | 0 |
| `+Y` | 4.143 | 2.961 | **0.71** | **7/20** |
| `−Y` | 3.299 | 2.864 | 0.87 | **4/20** |
| `+Z` | 5.138 | 2.797 | **0.54** | 0 |
| `−Z` | 5.137 | 2.871 | 0.56 | 0 |

Pojedyncza skala nie pokryje zakresu 0.54–0.90. Istotniejsze: **robosuite sam
jest anizotropowy i niesymetryczny** (`+Y` 4.14 wobec `−Y` 3.30 przy identycznym
module komendy), więc wzmocnienie zależy od konfiguracji ramienia, a nie jest
stałą mechanizmu. Wektor poprawek dopasowany w pozie home będzie błędny gdzie
indziej. To przesuwa rozwiązanie od strojenia skal w stronę zmiany regulatora
albo fine-tune'u polityki na zmierzonym mechanizmie.

Osobny problem: obrót wokół `Y` generuje **błędy IK w 4 do 7 przypadkach na 20**.
Ramię wchodzi w konfiguracje, których MoveIt nie rozwiązuje, a pętla traci wtedy
krok sterowania.

**Wynik: B2 NIE jest domknięte.** Po kalibracji ucieczka zmalała z 0.61 m do
plateau ok. 0.21 m, ale polityka nadal nie zbiega: `align` stoi w przedziale
80–89° zamiast schodzić do ~24°. Pozostała rozbieżność to **anizotropia
obrotu** — `rot +X` ma 1.01×, ale `rot +Y` tylko ok. 0.75×. Dopasowanie
wzmocnień po jednej osi nie wystarcza, bo mechanizm IK plus regulator
położenia stawów ma inną strukturę sprzężeń niż impedancyjny OSC, na którym
polityka była trenowana.

Kierunki do sprawdzenia: przemiatanie `tfs` poniżej 0.02 łącznie z korektą
skali per oś obrotu; zastąpienie ścieżki IK plus trajektoria regulatorem
bliższym impedancyjnemu; fine-tune polityki w Webots na zmierzonym mechanizmie.

### 6.29. Warstwa B2 — MoveIt Servo: anizotropia usunięta, ale sufit tempa zostaje (2026-09-11)

Rozdz. 6.28 zostawił dwa objawy: wzmocnienie zależne od osi (0.54–0.90) i błędy
IK przy obrocie wokół `Y`. Hipoteza mówiła, że winien jest regulator, który
interpoluje w przestrzeni **stawów**, przez co to samo zadanie kartezjańskie
kosztuje różną liczbę radianów zależnie od jakobianu. Test: przełączyć ścieżkę
wykonawczą na MoveIt Servo, czyli regulator kartezjański liczący jakobian sam.

#### Co się zmieniło w moście, a co nie

**Ścieżka obserwacji jest nietknięta.** `obs.py`, `build()`, `home()` i `act()`
działają identycznie. Zmieniają się wyłącznie trzy ostatnie linie pętli:
`action_to_target` + `solve_ik` + `send` zastąpione przez `send_twist()`.
Obie drogi żyją obok siebie za flagą `--servo`, więc są porównywalne.

Servo okazało się wyjątkowo dobrze dopasowane do tego zastosowania:
`command_in_type: "unitless"` przyjmuje twist w zakresie `[-1, 1]`, czyli
dokładnie w formacie akcji polityki, a `robot_link_command_frame` to
`gen3_base_link`, ta sama ramka, w której polityka dostaje obserwację.
Znikają `GRIP_OFFSET`, składanie kwaternionów i cała klasa błędów
z `action_to_target`.

#### Hipoteza potwierdzona

| | droga przez IK | przez Servo |
|---|---|---|
| rozrzut wzmocnienia po 6 osiach | 0.54–0.90 | **0.04–0.05, jednakowe** |
| błędy IK przy obrocie `Y` | 4–7 na 20 | brak |

Anizotropia **znika całkowicie**. Wszystkie sześć osi, oba znaki, translacja
i obrót mają to samo wzmocnienie. Diagnoza z 6.28 była poprawna.

#### Czego nie udało się domknąć: sufit tempa

Wzmocnienie jest jednorodne, ale dwudziestokrotnie za małe. Sprawdzono trzy
dźwignie, wszystkie zmierzone:

**Wielkość komendy.** Nasyca się. Translacja osiąga 0.32× tempa robosuite przy
komendzie bezwymiarowej 0.2 i dalej nie rośnie (0.4 i 0.6 dają tyle samo).
Obrót powyżej zakresu `[-1, 1]` daje ruch **zerowy**.

**Limity przyspieszeń** w `moveit_joint_limits.yaml` (repozytorium MASKOR):
podniesione z `0.7` na `10.0` dla stawów 1–3 i z `1.5` na `15.0` dla 4–6.
Pomogło przy małych komendach (0.05× → 0.17×), ale **sufitu nie ruszyło**.
Zmiana została w pliku, bo poprawia odpowiedź, i jest tam opisana.

**`publish_period`** w `moveit_servo_config.yaml`: przemiecione 0.05 / 0.10 /
0.15 / 0.25 dało 0.31–0.32× w każdym przypadku. Parametr **nie jest dźwignią**;
zmiana cofnięta do oryginalnego 0.01 z adnotacją.

Sufit końcowy: **translacja 0.32×, obrót 0.10×** tempa robosuite.

#### Właściwa przyczyna

Obie ścieżki — IK plus trajektoria oraz Servo — publikują na **ten sam**
`/kinova_joint_trajectory_controller/joint_trajectory`. Servo ma go wpisanego
jako `command_out_topic`. Ten regulator realizuje około jednej trzeciej
zadanego punktu, zanim następny go nadpisze. Wąskim gardłem nie jest więc ani
IK, ani Servo, ani limity stawów, tylko **regulator położenia stawów
w Webots**. Servo naprawiło kształt odpowiedzi, nie jej amplitudę.

#### Zachowanie polityki

| ścieżka | dystans po 40 krokach | `align` | charakter |
|---|---|---|---|
| IK, `tfs = 0.05` | 0.045 → **0.614** | 89.9° → 55° | oscylacja co drugi krok |
| IK, skalibrowana | 0.045 → 0.21 (plateau) | stoi 80–89° | plateau |
| **Servo** | 0.045 → **0.077** | 89.9° → 86.1° | **gładko, bez oscylacji** |

Przez Servo ruch jest po raz pierwszy stabilny i kierunkowo sensowny. Polityka
nie rozbiega się, tylko **nie zdąża** — przy 0.10× tempa obrotu nie wykona
w 40 krokach manewru, który w robosuite zajmuje 30.

#### Wnioski i dalsze kroki

Strojenie stałych jest wyczerpane. Zostają dwie drogi:

- **Zmienić regulator w Webots.** Sterowanie prędkością albo momentem zamiast
  pozycją stawów usunęłoby sufit u źródła. To ingerencja w sterownik MASKOR.
- **Fine-tune polityki z randomizacją wzmocnień.** W robosuite, nie w Webots —
  w Webots nie da się policzyć nagrody, bo przyciski nie mają fizyki (6.24),
  a 1M kroków przy 20 Hz w czasie rzeczywistym to ok. 14 h zegara. Wariant
  sensowny: opakowanie skalujące osie akcji przed OSC, ze współczynnikami
  **losowanymi na epizod** (np. 0.3–1.2), żeby polityka uodporniła się na
  mechanizm zamiast dopasowywać do jednej jego wersji.

Druga droga jest wartościowa niezależnie od pierwszej, bo prawdziwy Gen3 ma
jeszcze inne API sterowania kartezjańskiego, a Webots jest tylko platformą
walidacyjną.

### 6.30. Randomizacja wzmocnienia mechanizmu — uzasadnienie i przebieg (2026-09-11)

#### Problem, który to wywołał

Rozdziały 6.28 i 6.29 zmierzyły, że **ten sam wektor akcji daje w Webots ułamek
ruchu, który daje w robosuite**:

| ścieżka wykonawcza | translacja | obrót |
|---|---|---|
| IK + `JointTrajectory` | 0.54–0.90×, **różne dla każdej osi** | 0.54–0.90× |
| MoveIt Servo | 0.32× | 0.10× |

Polityka wystawia akcje identyczne z robosuite, zweryfikowane co do 0.001, więc
to nie jest usterka mostu ani obserwacji. To **rozbieżność mechanizmu**. Sufit
leży w `JointTrajectoryController` Webotsa, na który publikują obie ścieżki.

#### Dlaczego to nie jest „to samo, tylko wolniej"

Gdyby wszystkie osie spowolniły jednakowo, a obserwacja nie zawierała
prędkości, trajektoria w przestrzeni stanu byłaby identyczna, tylko rozciągnięta
w czasie — polityka jest bezpamięciowa (`MlpPolicy`, bez rekurencji), więc
w danym stanie zawsze wystawia to samo. Oba warunki są jednak złamane:

1. **`obs[18:24]` to prędkości stawów.** Przy 0.32× są trzykrotnie mniejsze niż
   cokolwiek, co polityka widziała w treningu — obserwacja jest poza rozkładem.
2. **Spowolnienie jest anizotropowe** (0.32 wobec 0.10). Zmienia się przez to
   *kształt* trajektorii, nie samo tempo.

Wydłużenie horyzontu tego nie naprawi i nie jest rozwiązaniem.

#### Dlaczego randomizacja, a nie dopasowanie do Webotsa

Alternatywą było zmierzyć wzmocnienie Webotsa i wstawić je na sztywno, albo
przebudować sterownik w `webots_ros2_driver` na prędkościowy. Odrzucone
z trzech powodów:

1. **Wzmocnienie nie jest stałą mechanizmu, tylko funkcją konfiguracji ramienia.**
   Dowód jest w samym robosuite: `+Y` daje 4.14 °/krok, a `−Y` tylko 3.30 °/krok
   przy identycznym module komendy (6.28). Stała dopasowana w pozie home będzie
   błędna gdzie indziej, a polityka przejeżdża przez wiele konfiguracji.
2. **Webots jest platformą walidacyjną, nie celem.** Prawdziwy Gen3 ma własne API
   kartezjańskie Kinovy i jeszcze inny mechanizm. Dostrojenie do sterownika
   MASKOR nie przeniesie się na sprzęt.
3. **Randomizacja daje politykę obojętną na mechanizm** — jedyną rzecz, która
   faktycznie jedzie do laboratorium. To także jedyny wariant, który adresuje
   ryzyko, że sprzęt okaże się jeszcze inny niż oba symulatory.

Naprawa sterownika pozostaje otwarta jako osobny wątek, ale jest inwestycją
w środowisko, które i tak zostanie porzucone.

#### Implementacja

`ButtonBoardEnv` dostał dwa parametry: `gain_noise_lin` i `gain_noise_rot`,
każdy jako `(lo, hi)` albo `None`. Mnożnik jest losowany **na epizod
i osobno dla każdej z sześciu osi**, żeby odwzorować zmierzoną anizotropię.
Przyjęty zakres to `[0.3, 1.2]`: dolna granica pokrywa zmierzone 0.32 z zapasem,
a górna przekracza 1.0, żeby polityka nie nauczyła się, że mechanizm jest zawsze
wolniejszy od nominalnego.

Jedna pułapka warta odnotowania. Nagroda zawiera `ACTION_COST` i
`action_rate_cost`, liczone na akcji przekazanej do `step()`. Gdyby skalowanie
weszło przed tym punktem, randomizacja zmieniałaby **reward**, a bieg mieszałby
dwie zmienne naraz. Dlatego `step()` zapamiętuje `self._raw_action` przed
przeskalowaniem, a `reward()` liczy karę na akcji polityki, nie na akcji podanej
regulatorowi. Skalowanie widzi wyłącznie mechanizm.

Do pomiaru `compare_models.py` dostał flagę `--gain`, która ustawia zakres
zdegenerowany do punktu, czyli wzmocnienie **stałe** zamiast losowanego.

#### Kryterium sukcesu

Podwójne i oba trzeba zmierzyć przy `h = 1500`:

- **utrzymanie:** przy `gain = 1.0` skuteczność zostaje w okolicy 76/100.
  Inaczej zapłaciliśmy odpornością za jakość;
- **zysk:** przy `gain = 0.5` i `gain = 0.32` rośnie wyraźnie wobec modelu
  bazowego.

Sam zysk bez utrzymania **nie jest sukcesem**. Zadanie jest teraz trudniejsze od
bazowego, bo dochodzi sześć osi randomizacji, więc spadek o kilka punktów przy
`gain = 1.0` jest oczekiwany.

#### Model bazowy przy obniżonym wzmocnieniu — pomiar „przed"

`ft_sc1_v23_long/ppo_board_final.zip` (15.91M), 30 epizodów, `h = 1500`,
wzmocnienie **stałe** (`compare_models --gain`), bez randomizacji:

| wzmocnienie | sukcesy | porażki wg wciśniętych `[0..4]` | zacięcia | wciśnięcia btn0–btn4 |
|---|---|---|---|---|
| 1.00 | **19/30** | `[0, 5, 4, 2, 0]` | 3 | 21 / 21 / 21 / 26 / 25 |
| 0.50 | **2/30** | `[18, 6, 1, 3, 0]` | 3 | 5 / 5 / 6 / 3 / 8 |
| 0.32 | **0/30** | `[30, 0, 0, 0, 0]` | **16** | 0 / 0 / 0 / 0 / 0 |

Wiersz `1.00` zgadza się z udokumentowanymi 76/100 w granicach błędu
próbkowania przy n = 30.

**Wiersz `0.32` jest kluczowy, bo to dokładnie wzmocnienie, które dostarcza
MoveIt Servo w Webots** (6.29). Model bazowy nie wciska tam **ani jednego
przycisku w 30 epizodach**, a w 16 z nich się zacina. To zamyka wątek
diagnostyczny warstwy B2: zachowanie obserwowane w Webots było w całości
skutkiem rozbieżności mechanizmu, a nie usterką mostu, obserwacji czy polityki.

Degradacja jest gwałtowna, nie stopniowa: 19 → 2 → 0 przy spadku wzmocnienia
1.00 → 0.50 → 0.32. Potwierdza to argument z początku rozdziału, że nie jest to
„to samo, tylko wolniej" — gdyby było, dłuższy horyzont by wystarczył.

Uwaga metodyczna: powyższe to trzy **niezależne** przebiegi, nie epizody
parowane. Finalne porównanie bazy z modelem po fine-tunie zostanie zrobione
`compare_models` z obydwoma modelami naraz, osobno dla każdego wzmocnienia,
żeby dostać test istotności na identycznych warunkach początkowych.

#### Stan biegu (PRZERWANY na 18.71M)

Bieg zatrzymany ręcznie po 2.80M z zaplanowanych 3.00M kroków dodatkowych
(4 h 40 min, tempo ok. 100k / 9–10 min). Zapisane:

- ostatni checkpoint `logs/board_6dof/gain_rnd/checkpoints/ppo_board_18710624_steps.zip`
- `logs/board_6dof/gain_rnd/best_model/best_model.zip` — wybierany po nagrodzie
  ewaluacyjnej, przy czym środowisko ewaluacyjne **dziedziczy randomizację
  wzmocnienia**, więc selekcja idzie pod odporność, nie pod nominał

Nagroda ewaluacyjna trzymała się w przedziale −2.4 … +0.2. To **nie jest sygnał
awarii**: bieg bazowy `ft_sc1_v23_long` miał tę samą skalę (−1.5 … +0.7,
końcowo 0.6). W tym środowisku nagroda jest zdominowana przez kary i shaping,
które prawie się znoszą, więc jej wartość bezwzględna nie mówi nic
o skuteczności — oceniać wyłącznie przez `compare_models`.

Wznowienie:

```
python training/train_board.py \
    --config training/configs/ppo_board_6dof_gain_rnd.yaml \
    --resume logs/board_6dof/gain_rnd/checkpoints/ppo_board_18710624_steps.zip \
    --timesteps 200000
```

Bieg dokończony 2026-09-12 — wyniki poniżej.
#### WYNIK: NEGATYWNY (2026-09-12)

Bieg dokończony do **18.92M** (2.81M kroków dodatkowych, 4 h 58 min łącznie).
Porównanie z bazą na **epizodach parowanych**, `compare_models` z obydwoma
modelami naraz, po 40 epizodów na poziom, `h = 1500`:

| wzmocnienie | baza | po fine-tunie | tylko baza | tylko nowy | p (pary) |
|---|---|---|---|---|---|
| 1.00 | **24/40** | 18/40 | 9 | 3 | 0.146 |
| 0.50 | 2/40 | 3/40 | 1 | 2 | 1.000 |
| 0.32 | 0/40 | 0/40 | 0 | 0 | 1.000 |

**Oba kryteria niespełnione.** Utrzymanie: spadek 24 → 18 przy nominale, nieistotny
statystycznie, ale kierunkowo jednoznaczny. Zysk: brak przy obniżonym wzmocnieniu.

#### Realna szkoda to kolizyjność, nie skuteczność

| wzmocnienie | kroki kolizji/ep, baza | po fine-tunie | |
|---|---|---|---|
| 1.00 | 3.0 | 28.1 | **9.4×** |
| 0.50 | 13.1 | 32.8 | 2.5× |
| 0.32 (h = 4000) | 53.1 | 175.0 | **3.3×** |

To jest regresja dużo wyraźniejsza od zmiany skuteczności i **dyskwalifikuje ten
model do użycia na sprzęcie**. Interpretacja: polityka nauczyła się napierać
mocniej, żeby radzić sobie z możliwym niskim wzmocnieniem, i przy wzmocnieniu
nominalnym wjeżdża w tablicę. Jest to dokładnie ta klasa zachowania, którą
`collision_cost = 5.0` miał tłumić (6.18) — randomizacja ją przywróciła.

#### Błąd w kryterium i jego sprostowanie

Pierwotne kryterium mierzyło przy `h = 1500` także poziom `0.32`. To było
**nieosiągalne z definicji**: baza kończy sukcesem w 561 krokach przy wzmocnieniu
nominalnym, więc przy `0.32` ten sam manewr wymaga ok. 1753 kroków. Liniowość
potwierdzona niezależnie — model po fine-tunie przy `0.5` kończył w 1174 krokach
wobec przewidywanych 1122.

Pomiar powtórzono przy `h = 4000`. **Hipoteza o horyzoncie obalona**: przy `0.32`
nadal 0/25 dla obu modeli. Awaria przy niskim wzmocnieniu nie jest brakiem czasu,
tylko trwałym zacięciem — baza zacina się w 19 z 25 epizodów i nie wciska
**ani jednego** przycisku. Potwierdza to tezę z początku rozdziału: to nie jest
„to samo, tylko wolniej", tylko inna pętla sprzężenia.

#### Co jednak zadziałało

Skuteczność pełnej sekwencji przesłania efekt widoczny w postępie cząstkowym:

| pomiar | model | wciśnięte/ep | zacięcia |
|---|---|---|---|
| `h=4000`, gain 0.50 | baza | 1.08 | 2/25 |
| | po fine-tunie | **1.80** | 2/25 |
| `h=4000`, gain 0.32 | baza | 0.00 | **19/25** |
| | po fine-tunie | **0.12** | **10/25** |

Randomizacja **zmniejszyła zacinanie się o połowę** przy `0.32` (19 → 10) i
podniosła liczbę wciśniętych przycisków o ok. 70 % przy `0.5`. Mechanizm działa
w zamierzonym kierunku, tylko efekt jest za słaby, żeby domknąć sekwencję
pięciu przycisków, a koszt w kolizyjności go przewyższa.

#### Wnioski

1. **Model `gain_rnd` NIE zastępuje `ft_sc1_v23_long`.** Model na sprzęt pozostaje
   bez zmian.
2. **Randomizacja wzmocnienia jako taka nie jest obalona** — jej sygnatura jest
   widoczna w postępie cząstkowym i w zacięciach. Obalona jest ta konkretna
   konfiguracja: zakres `[0.3, 1.2]` przy budżecie 2.8M i niezmienionej
   `collision_cost`.
3. **Prawdopodobna przyczyna porażki: zakres za szeroki na ten budżet.**
   Sześć osi losowanych niezależnie w przedziale niemal czterokrotnym to bardzo
   szerokie zadanie; 2.8M kroków to ok. 18 % budżetu, za który powstała baza.
4. **Do sprawdzenia, gdyby wracać do tematu:** węższy zakres (np. `[0.6, 1.1]`),
   curriculum rozszerzające go w trakcie, oraz podniesienie `collision_cost`
   razem z randomizacją — bez tego polityka kupuje odporność napieraniem.
   To jednak **czwarta próba naprawy zachowania przez zmianę treningu** i
   obowiązuje zasada z 6.20: kolejne podejście wymaga uzasadnienia mocniejszego
   niż „spróbujmy innych liczb".
5. **Wniosek dla sim2real jest mimo wszystko pozytywny:** zmierzono, że polityka
   trenowana na jednym mechanizmie traci całą skuteczność przy wzmocnieniu 0.32
   i że nie da się tego nadrobić czasem. To jest wynik sam w sobie i argument za
   tym, żeby na sprzęcie **zmierzyć wzmocnienie mechanizmu przed uruchomieniem
   polityki**, a nie po.

Artefakty: `logs/board_6dof/gain_rnd/ppo_board_final.zip` (18.92M),
`best_model/best_model.zip`, CSV-e porównań w katalogu roboczym sesji.


### 6.31. Limity prędkości stawów z firmware Gen3 — najtwardsze ograniczenie transferu (2026-09-12)

#### Skąd liczby

Limitów na sprzęcie nie dało się zmierzyć samodzielnie, więc przyjęto oficjalne
repozytorium producenta jako źródło. To **założenie**, nie pomiar.

> **SPROSTOWANIE (2026-09-12, w trakcie rozdziału).** Pierwsza wersja tego
> rozdziału przyjmowała **1.3963 / 1.2218 rad/s** z PR #241, który zmieniał tagi
> `<velocity>` w `gen3_macro.xacro`. **Wartości wiążące są inne i tkwią
> w osobnym pliku**, `kortex_description/arms/gen3/6dof/config/joint_limits.yaml`:
>
> ```
> maximum_velocities:    0.8727 rad/s  × 6   (= 50 °/s, zgodne z kartą katalogową)
> maximum_accelerations: 1.0 (j1–j3), 10.0 (j4–j6) rad/s²
> ```
>
> Oraz `twist_limits.yaml` dla sterowania kartezjańskiego:
> ```
> maximum_linear_velocity: 0.5      maximum_angular_velocity: 1.74
> maximum_linear_acceleration: 0.4  maximum_angular_acceleration: 0.4
> ```
>
> Prawdziwy limit prędkości jest więc **1.6× ciaśniejszy**, jednakowy dla
> wszystkich sześciu stawów. Wszystkie liczby w sekcjach „Polityka żąda więcej"
> i „Krok 1/2" poniżej pochodzą ze **starego, zbyt luźnego założenia** i są
> **dolnym oszacowaniem** problemu. Pomiar na właściwych limitach niżej.

Dla porządku, źródłem pierwotnego (błędnego) założenia był PR #241
„Joint limits modifications", którego uzasadnienie brzmi: *„The new joint limits
reflect the ones in the robot's firmware. This way we avoid the mismatch between
the virtual representation and reality."*

#### Polityka żąda więcej, niż robot potrafi

Pomiar na modelu 15.91M, 8 epizodów, 6324 kroki:

| staw | mediana | p95 | max | limit firmware | % kroków ponad limitem |
|---|---|---|---|---|---|
| j1 | 0.053 | 0.380 | 0.560 | 1.3963 | 0.0 % |
| j2 | 0.125 | 0.666 | 2.072 | 1.3963 | 0.1 % |
| j3 | 0.292 | 1.287 | 3.322 | 1.3963 | 3.7 % |
| j4 | 0.524 | 1.819 | 12.300 | 1.2218 | 16.4 % |
| j5 | 0.342 | 1.458 | 3.635 | 1.2218 | 8.6 % |
| j6 | 1.165 | 2.471 | 10.092 | 1.2218 | **46.9 %** |

**W 59.2 % kroków co najmniej jeden staw przekracza możliwości sprzętu.**
Rozkład problemu jest wymowny: ramię mieści się prawie zawsze, **nadgarstek nie**.
To te same stawy, które realizują obrót, i zgadza się to z sim2sim, gdzie w Webots
obrót padał mocniej niż translacja (6.29).

Uwaga niezależna: Webots MASKOR miał limity `3.0` / `4.0` rad/s, czyli **2.1×
luźniejsze niż firmware**. Symulator walidacyjny kłamał na naszą korzyść.
Sprowadzone do wartości firmware'u (krok 3 poniżej).

#### Implementacja: poprawka wierności, nie kara

MuJoCo nie zna limitu prędkości, więc `ButtonBoardEnv._pre_action()` nakłada
saturację na `qvel` przed każdym krokiem fizyki. Modeluje to napęd, który
po prostu nie potrafi szybciej.

**To NIE podpada pod zakaz z 6.20.** Nagroda jest nietknięta, nic nie jest
randomizowane. Jest to ta sama klasa zmiany co sprowadzenie geometrii z Webotsa
(6.24): symulator zaczyna odwzorowywać ograniczenie, które na sprzęcie istnieje.

Parametr `joint_vel_limits` (6 wartości albo `None`, domyślnie `None`),
w `compare_models.py` flaga `--hw-limits`.

#### ⚠ NIESZCZELNA IMPLEMENTACJA LIMITU — wykryta 2026-09-13

Test mechanizmu kurikulum (6.32) wykazał, że opisana wyżej saturacja `qvel`
**przed** krokiem fizyki **nie egzekwowała limitu na nadgarstku**. Przy limicie
`0.8727` i szarpanej akcji obserwowane maksima prędkości:

| staw | j2 | j3 | j4 | j5 | j6 |
|---|---|---|---|---|---|
| max \|qvel\| [rad/s] | 0.887 | 1.055 | **2.430** | **2.263** | **2.524** |
| krotność limitu | 1.02 | 1.21 | **2.78** | **2.59** | **2.89** |

Przyczyna: stawy nadgarstka mają `armature 0` i `damping 0.01` (`j1` ma
`armature 5.0`), a robosuite z `lite_physics=True` całkuje w `step2`. Lekkie
ogniwo pod momentem OSC przyspieszało w jednym podkroku 2 ms ponad limit,
a pozycja przesuwała się z tą przekroczoną prędkością.

**Skutek dla wyników tego rozdziału.** Wszystkie pomiary i treningi Z NAŁOŻONYM
limitem odbyły się w obwiedni **luźniejszej niż firmware na nadgarstku**:
Krok 1 (27/40 → 1/40), Krok 2 (`hwlim`), scratch `hwlim_s1`, diagnostyka
`depress_ft` (docisk, przemiatanie horyzontu, wzmocnienia 1.00× / 0.60× / 0.69×).
Są to wyniki **optymistyczne** względem prawdziwego sprzętu — rzeczywista
degradacja na nadgarstku jest najpewniej co najmniej tak duża.

**Nieskażone** pozostają pomiary rozkładu prędkości polityki bez limitu
(81.2 % kroków ponad 0.8727, 88 % ponad limitem przyspieszeń), bo nie używały
saturacji.

Naprawa i jej weryfikacja: rozdz. 6.32.

#### Krok 1 — ile kosztują limity (model 15.91M)

Pary, n = 40, h = 1500, ten sam model, jedyna różnica to limity:

| | bez limitów | z limitami |
|---|---|---|
| sukcesy | **27/40** | **1/40** |
| wciśnięte/ep | 4.05 | 1.18 |
| kroki kolizji/ep | 16.2 | **75.5** |
| misuse/ep | 0.2 | 13.2 |

26 epizodów przegranych wyłącznie przez limity, 0 odwrotnie, **p = 2.98e-08**.

Drugi objaw jest groźniejszy od spadku skuteczności. **Kolizyjność rośnie
4.7×, a użycie ramienia zamiast palców 66×.** Polityka obcinana w prędkości
przewraca się na tablicę — na sprzęcie byłoby to uderzanie w omniboard.

#### Krok 2 — fine-tune pod limitami: WYNIK MIESZANY

`ppo_board_6dof_hwlim.yaml`, 3M kroków od 15.91M do 18.91M, 5 h.
Zmienna niezależna: wyłącznie `joint_vel_limits`.

| pomiar | baza | hwlim | tylko baza | tylko nowy | p (pary) |
|---|---|---|---|---|---|
| limity HW, h = 1500 | 1/40 | 5/40 | 1 | 5 | 0.22 |
| limity HW, h = 2000 | 3/40 | 5/40 | 3 | 5 | 0.73 |
| **bez limitów, h = 1500** | **27/40** | **6/40** | 22 | 1 | **5.7e-06** |

| pomiar | kolizje baza | kolizje hwlim | |
|---|---|---|---|
| limity HW, h = 1500 | 75.5 | **2.8** | **27×** |
| limity HW, h = 2000 | 127.2 | **6.7** | 19× |
| bez limitów, h = 1500 | 16.2 | 8.5 | 1.9× |

**Kryterium skuteczności NIESPEŁNIONE.** Założono > 40/100 przy h = 1500;
wyszło 12/100, nieistotnie wobec bazy. Własne kryterium stopu z nagłówka
configu („jeśli w ciągu 1.5M sukces nie przekroczy ~20 %, fine-tune nie
wystarcza") **zostało przekroczone w dół** po pełnych 3M.

**Kryterium kolizyjności SPEŁNIONE z dużym zapasem.** Założono < 20 kroków/ep;
wyszło **2.8**, czyli 27× lepiej od bazy i lepiej niż baza osiąga nawet
w warunkach nominalnych (16.2). To najsilniejsza poprawa kolizyjności w całym
projekcie. Misuse spadł z 13.2 na 0.1 kroku/ep, zacięcia z 10/40 na 4/40.

**Koszt: utrata nominału.** Bez limitów nowy model daje 6/40 wobec 27/40 bazy
(p = 5.7e-06). Model jest wyspecjalizowany pod obwiednię sprzętową i nie
zastępuje bazy w warunkach nieograniczonych.

**Horyzont nie jest wąskim gardłem.** Przy h = 2000 nowy model nadal daje 5/40,
a sukcesy przychodzą w 1130 krokach. Porażki to 15 epizodów z zerem wciśnięć —
to nie brak czasu, tylko nieumiejętność.

#### Krok 3 — Webots sprowadzony do firmware'u

`webots_ros2_spot/resource/moveit_joint_limits.yaml`: prędkości `3.0` → `1.3963`
(j1–j3) i `4.0` → `1.2218` (j4–j6). Przyspieszenia zostawione podniesione
(10.0 / 15.0), bo firmware ich nie podaje, a oryginalne 0.7 / 1.5 dławiły Servo.

Skutek uboczny: Webots będzie jeszcze wolniejszy niż zmierzone `0.32×`.
Jest to poprawne. Sim2sim w Webots przestaje być testem skuteczności, a zostaje
testem percepcji, mostu obserwacji i nadzorcy — czyli tym, po co go wybrano.

#### Pomiar na WŁAŚCIWYCH limitach (0.8727 rad/s)

Ten sam model 15.91M, te same 6324 kroki, limit `0.8727` dla wszystkich stawów:

| staw | mediana | p95 | max | % ponad 0.8727 | % ponad starym założeniem |
|---|---|---|---|---|---|
| j1 | 0.053 | 0.380 | 0.560 | 0.0 % | 0.0 % |
| j2 | 0.125 | 0.666 | 2.072 | 1.7 % | 0.1 % |
| j3 | 0.292 | 1.287 | 3.322 | 14.2 % | 3.7 % |
| j4 | 0.524 | 1.819 | 12.300 | 31.1 % | 16.4 % |
| j5 | 0.342 | 1.458 | 3.635 | 18.0 % | 8.6 % |
| j6 | 1.165 | 2.471 | 10.092 | **64.5 %** | 46.9 % |

**81.2 % kroków** przekracza limit prędkości, nie 59.2 %.

Przyspieszenia są jeszcze gorsze. Wobec `1.0` (j1–j3) i `10.0` (j4–j6) rad/s²:

| staw | mediana a | p95 | limit | % ponad |
|---|---|---|---|---|
| j1 | 0.153 | 1.40 | 1.0 | 7.4 % |
| j2 | 1.635 | 6.97 | 1.0 | 63.9 % |
| j3 | 4.847 | 21.99 | 1.0 | **76.9 %** |
| j4 | 9.037 | 40.43 | 10.0 | 47.3 % |
| j5 | 7.556 | 34.10 | 10.0 | 41.9 % |
| j6 | 12.298 | 58.06 | 10.0 | 56.0 % |

**88.0 % kroków** wychodzi poza obwiednię przyspieszeń. Mediana na `j3` jest
prawie **5× ponad limitem**.

**Świadoma decyzja: przyspieszeń NIE nakładamy w treningu.** Prędkość `0.8727`
zgadza się z kartą katalogową (50 °/s), więc jest niemal na pewno limitem
firmware'u. Wartości `1.0` / `10.0` pochodzą z tego samego pliku, ale równie
dobrze mogą być limitami **planera MoveIt** używanymi do parametryzacji czasowej,
a nie twardym ograniczeniem napędu. Nie ograniczamy symulatora liczbą, której
pochodzenia nie jesteśmy pewni — `joint_acc_limits` istnieje w `ButtonBoardEnv`,
ale domyślnie jest wyłączone. **Do zweryfikowania na sprzęcie.**

#### Scratch z limitami (etap 1) i diagnoza: co dokładnie się psuje

`ppo_board_6dof_hwlim_s1.yaml`, **od zera**, 8M kroków, kurikulum jak w 6.17
(chwytak swobodny, bez bramki palcowej, `collision_cost` 0.5,
`action_rate_cost` 0), plus `log_std_max` od początku. Wynik: **0/40**, zero
wciśnięć na wszystkich pięciu przyciskach, 36/40 epizodów zakończonych
zacięciem, gładkość 0.28 (akcja prawie stała).

Trzy pomiary diagnostyczne, wszystkie na `depress_ft` — modelu, który **bez
limitów** działa (80/100 w 6.17), żeby oddzielić „zadanie niewykonalne" od
„ten bieg się nie nauczył".

**1. Wykonalność docisku.** Ten sam model, 12 epizodów, protokół etapu 1:

| | bez limitów | z limitami |
|---|---|---|
| min. dystans do celu, mediana | 5.0 mm | 17.5 mm |
| max. głębokość wciśnięcia, mediana | 13.1 mm | **12.4 mm** |
| epizody z dotknięciem grzybka | 12/12 | **12/12** |
| epizody z przekroczeniem progu 12 mm | 11/12 | **8/12** |

**Docisk pod limitami DZIAŁA.** Hipoteza o fizycznej niewykonalności jest
obalona. Nie ma potrzeby modelowania płytki dociskowej ani modyfikacji chwytaka.

**2. Czy to brak czasu.** Przemiatanie horyzontu, 20 epizodów na punkt:

| limity | h = 1000 | h = 2000 | h = 3000 | h = 4000 |
|---|---|---|---|---|
| brak — sukcesy | 10/20 | 15/20 | 17/20 | **18/20** |
| brak — wciśnięte/ep | 3.70 | 4.60 | 4.70 | 4.80 |
| 0.8727 — sukcesy | 0/20 | 0/20 | 0/20 | **0/20** |
| 0.8727 — wciśnięte/ep | 0.55 | 0.70 | 0.75 | **0.75** |

**Hipoteza o horyzoncie OBALONA.** Czterokrotne wydłużenie epizodu nie daje nic,
liczba wciśnięć nasyca się na 0.75 z pięciu. Bez limitów ta sama polityka
dochodzi do 4.80.

> Uwaga metodologiczna: wzorzec wciśnięć per przycisk (`7, 2, 1, 2, 2`) wyglądał jak
> sygnatura kończącego się czasu i tak go początkowo odczytano. Był to błędny
> odczyt — to sygnatura **zatrzymania po pierwszym przycisku**. Rozstrzygnął
> dopiero pomiar, nie interpretacja.

**3. Gdzie się psuje.** Ramię wciska **pierwszy** przycisk w ok. 70 % epizodów,
po czym nie przechodzi do następnego. Przejście między gniazdami wymaga dużego
obrotu nadgarstka, a obrót jest jedyną osią realnie dławioną: zmierzone
wzmocnienie translacji pod limitami to **1.00×**, obrotu `+X` **0.60×**,
obrotu `+Y` **0.69×**.

Zadanie nie jest więc ani niewykonalne, ani wolniejsze — jest **rozspójnione**:
pojedynczy docisk zachowuje się prawie normalnie, a sekwencja pięciu przycisków
się rozpada.

#### Błąd metodyczny do zapamiętania

Trzynastogodzinny trening `hwlim_s1` został uruchomiony **przed** testem
wykonalności, który zajął dziesięć minut. Test pokazał, że zadanie jest
wykonalne, ale inaczej trudne, niż zakładano — a więc bieg był oparty na
niesprawdzonej przesłance.

**Reguła:** przed treningiem w zmienionej obwiedni (limity, geometria, mechanizm)
najpierw przepuścić przez nią **znany dobry model** i zmierzyć, co się psuje.
Dopiero potem projektować bieg.

To jest rozszerzenie zasady z 6.30 o skalowaniu horyzontu — obie należą do
domyślnego protokołu pomiaru, a nie do opisu pojedynczego rozdziału.

#### Wnioski

1. **Limity prędkości stawów są najtwardszym zidentyfikowanym ograniczeniem
   transferu**, twardszym od geometrii i od szumu percepcji. Kosztują 27/40 → 1/40
   przy modelu, który poza nimi jest sprawny.
2. **Fine-tune nie wystarcza.** 3M kroków dało 12/100. Zadanie w obwiedni
   sprzętowej jest jakościowo inne, nie trudniejszą wersją tego samego.
3. **Ale kierunek jest słuszny** — dowodzi tego kolizyjność. Model wytrenowany
   z limitami jest w tej obwiedni **bezpieczny** (2.8 kroku/ep), podczas gdy baza
   jest niebezpieczna (75.5). Gdyby wybierać dziś model na sprzęt, to mimo
   niższej skuteczności `hwlim` jest jedynym, którym wolno ruszyć robota.
   > **SPROSTOWANIE (6.32, 2026-09-13):** wniosek nieważny. Kolizyjność 2.8 była
   > artefaktem nieszczelnego limitu przy błędnych wartościach 1.3963 / 1.2218.
   > Pod naprawionym limitem 0.8727 `hwlim` ma 0/40 sukcesów i **60.3** kroku
   > kolizji/ep — więcej niż baza. Żaden model nie nadaje się dziś na sprzęt.
4. **Następny krok to scratch Z limitami od zera**, nie kolejny fine-tune.
   Baza 15.91M niesie 16M kroków nawyków wyuczonych poza obwiednią i to one
   są kotwicą. Budżet rzędu 15M, tak jak przy bazie.
5. **Do rozważenia równolegle, strukturalnie**: obniżyć `output_max` dla obrotu
   w OSC z 0.5 rad. Skoro nadgarstek przekracza limit w 47 % kroków, polityka
   jest zachęcana do żądań, których nigdy nie dostanie. To zmiana przestrzeni
   akcji, nie nagrody.
6. **Założenie do zweryfikowania na sprzęcie:** że limity z PR #241 są
   rzeczywiście egzekwowane przez firmware i że `twist_controller` z `ros2_kortex`
   nie ma dodatkowych ograniczeń. Cały ten rozdział stoi na tym założeniu.

Artefakty: `logs/board_6dof/hwlim/ppo_board_final.zip` (18.91M),
`training/configs/ppo_board_6dof_hwlim.yaml`.
### 6.32. Kurikulum po limicie prędkości i naprawa nieszczelnego limitu (2026-09-13)

#### Plan biegu

Diagnoza z 6.31 wskazała problem **odkrycia**, nie zdolności: scratch z limitem
od zera dojeżdżał do przycisku, ale nigdy nie dociskał, bo shaping głębokości
odpala się dopiero po kontakcie z grzybkiem. Zamiast kolejnego scratchu —
fine-tune z `depress_ft` (umie dociskać) z limitem zacieśnianym liniowo
**2.0 → 0.8727 rad/s**. Zadanie i kryterium docisku bez zmian: robot ma się
wyrównać z osią wciskania i docisnąć. Świadomie odrzucono modelowanie płytki
dociskowej i szukanie tolerancji kątowej grzybka — oba to obejście własnego
modelu, a nie rozwiązanie zadania.

Config: `training/configs/ppo_board_6dof_hwlim_curr.yaml`.

#### Mechanizm kurikulum

`JointVelLimitCurriculum` w `train_board.py` zmienia limit w ośmiu podprocesach
po każdym rolloucie. Dwie pułapki, obie obsłużone jawnie:

- **`vec_env.set_attr` byłby cichym no-opem.** W gymnasium 0.29 `Wrapper`
  (np. `Monitor`) ma `__getattr__`, ale nie ma `__setattr__` — atrybut trafiłby
  na `Monitor`, a fizyka zostałaby bez zmian. Dlatego `RobosuiteGymWrapper`
  dostał publiczne `set_joint_vel_limits` / `get_joint_vel_limits`, wołane przez
  `env_method`, a callback **odczytuje wartość z powrotem** ze wszystkich
  podprocesów i przerywa trening wyjątkiem przy rozjeździe.
- **Wznowienie.** Postęp liczony od `num_timesteps` w chwili startu biegu,
  nie od zera. Test przy liczniku 10M: kolejne limity `2.0 → 1.4364 → 0.8727`,
  końcowy odczyt z obu podprocesów `0.8727`.

Środowiska treningowe startują z limitem luźnym, **ewaluacja od początku liczy
się przy docelowym 0.8727** — `best_model` ma być najlepszy w obwiedni sprzętowej.

#### Usterka wykryta przy teście: limit nie trzymał na nadgarstku

Ten sam test sprawdzał wpływ na fizykę i wykazał, że dotychczasowa saturacja
`qvel` przed krokiem fizyki przepuszczała nadgarstek do **2.9× limitu**
(tabela w 6.31). Przyczyny:

| czynnik | wartość |
|---|---|
| podkroków fizyki na krok sterowania | 25 (`0.05 / 0.002`) |
| ścieżka całkowania | `lite_physics = True` → `sim.step1()` / `sim.step2()` |
| integrator | `mjINT_EULER` (półjawny) |
| `j1`: armature / damping | 5.0 / 0.1 |
| `j4–j6`: armature / damping | **0.0 / 0.01**, moment ±9 N·m |

Nadgarstek bez armatury przyspieszał w jednym podkroku 2 ms ponad limit,
a pozycja przesuwała się z przekroczoną prędkością.

#### Naprawa: limit egzekwowany po całkowaniu

Hak na `sim.step` **i** `sim.step2` (przy `lite_physics` robosuite woła `step2`
— hak na samo `step` byłby martwy), instalowany w `_initialize_sim`, bo
`hard_reset` tworzy nowy obiekt `sim` przy każdym resecie:

```
q_prev, v_prev = qpos, qvel        # przed całkowaniem
orig_step()
v = clip(qvel, -L, +L)             # (+ opcjonalnie clip przyspieszenia)
qvel = v
qpos = q_prev + v · dt
```

Przy integratorze Euler (`q_{t+1} = q_t + v_{t+1}·dt`) jest to **dokładnie** ruch
z prędkością nie większą niż limit — także w pozycji, a nie tylko w odczycie.
Stawy ramienia to zawiasy z `qposadr == dofadr == [0..5]`, więc indeksy pozycji
i prędkości się pokrywają.

#### Weryfikacja

Szarpana akcja przez 300 kroków. Sprawdzane trzy miary, bo sam odczyt `qvel`
dałoby się „naprawić" korektą bez zmiany ruchu:

| warunek | obs | każdy podkrok | z przyrostu pozycji |
|---|---|---|---|
| brak limitu | 6.687 | 7.226 | 6.837 |
| limit 2.0, epizod 1 | 2.0000 | 2.0000 | 2.0000 |
| limit 2.0, epizod 2 (nowy `sim`) | 2.0000 | 2.0000 | 2.0000 |
| limit 0.8727, epizod 1 | 0.8727 | 0.8727 | 0.8727 |
| limit 0.8727, epizod 2 (nowy `sim`) | 0.8727 | 0.8727 | 0.8727 |
| zmiana 2.0 → 0.8727 w locie | 0.8727 | 0.8727 | 0.8727 |

Stawy dochodzą **równo** do limitu (np. `[0.58, 0.87, 0.87, 0.87, 0.87, 0.87]`),
a nie zamarzają — hak ogranicza ruch, nie blokuje go.

#### Konsekwencja metodyczna

Bez testu wpływu na fizykę 6-godzinny bieg wystartowałby na obwiedni, której
nie egzekwowaliśmy, a wszystkie wnioski z 6.31 „z limitami" poszłyby do pracy
jako fakty o sprzęcie. **Każdy nowy mechanizm ograniczający (limit, saturacja,
randomizacja) trzeba weryfikować na skutku fizycznym, nie na odczycie
parametru.** To trzecia reguła protokołu, obok skalowania horyzontu (6.30)
i przepuszczenia znanego modelu przez nową obwiednię (6.31).

Przed startem biegu powtarzane są skażone pomiary z 6.31 na naprawionym limicie
(wyniki niżej).

#### Powtórka skażonych pomiarów na naprawionym limicie

**Wzmocnienia osi** (25 kroków stałej akcji od pozy home, chwytak zamknięty,
bez szumu):

| ruch | bez limitu | skażony limit | **naprawiony 0.8727** | stosunek |
|---|---|---|---|---|
| translacja `+Z` [m/krok] | 0.0129 | 0.0129 | **0.0129** | **1.00** |
| obrót `+X` [°/krok] | 4.150 | 2.475 | **2.216** | **0.53** |
| obrót `+Y` [°/krok] | 3.508 | 2.419 | **2.066** | **0.59** |
| obrót `+Z` [°/krok] | 5.160 | — | **2.475** | **0.48** |

**Wykonalność docisku `depress_ft`** (12 epizodów, protokół etapu 1, h = 600):

| | bez limitu | skażony limit | **naprawiony 0.8727** |
|---|---|---|---|
| min. dystans do celu, mediana | 5.0 mm | 17.5 mm | **25.2 mm** |
| max. głębokość, mediana | 13.1 mm | 12.4 mm | **12.3 mm** |
| epizody z dotknięciem grzybka | 12/12 | 12/12 | **12/12** |
| epizody z przekroczeniem 12 mm | 11/12 | 8/12 | **7/12** |

**Diagnoza z 6.31 przetrwała naprawę jakościowo**, liczby są nieco gorsze —
zgodnie z oczekiwaniem, bo skażony limit był luźniejszy na nadgarstku:
translacja nietknięta, obrót spada do ~0.5×, a docisk pod prawdziwym limitem
**nadal działa**. To spełnia regułę protokołu dla najtrudniejszego punktu
kurikulum (0.8727), więc bieg został uruchomiony przed dokończeniem pozostałych
powtórek, które służą dokumentacji i ewentualnemu dostrojeniu harmonogramu.

**Bieg uruchomiony 2026-09-13:** `ppo_board_6dof_hwlim_curr.yaml`, `--resume
logs/board_6dof/depress_ft/ppo_board_final.zip` (14.23M), `--timesteps 4000000`,
katalog `logs/board_6dof/hwlim_curr`, TensorBoard `ppo_board_6dof_hwlim_curr_0`.

**Weryfikacja startu biegu (TensorBoard, pierwsze iteracje).** Log standardowy
jest pusty, bo `conda run` buforuje wyjście — potwierdzenie z plików zdarzeń.

| iteracja | krok | limit (log) | limit (wzór) | `train/std` | `approx_kl` | `clip_fraction` |
|---|---|---|---|---|---|---|
| rollout 1 | 14 242 240 | 1.9926 | 1.9926 | — | — | — |
| 1 | 14 258 624 | 1.9852 | 1.9852 | 1.000 | **1035** | **1.000** |
| 2 | 14 275 008 | 1.978 | 1.9779 | 1.000 | 0.020 | 0.135 |
| 3 | 14 291 392 | 1.970 | 1.9705 | 1.000 | 0.018 | 0.104 |

Wzór: `limit = 2.0 − frac · (2.0 − 0.8727)`, `frac = (krok − 14 225 856) / 2 500 000`.
Kurikulum liczy postęp od licznika wznowienia, a nie od zera, zgodnie z testem.

**Clamp `log_std` działa od pierwszej iteracji** — `train/std = 1.0`, podczas gdy
`depress_ft` miał `σ` od 8.7 do 26.6.

**Skok KL w iteracji 1 jest artefaktem, nie rozbieżnością.** `LogStdClamp` działa
na końcu rolloutu, więc pierwszy rollout zebrała jeszcze polityka z eksplodowanym
`σ ≈ 15`. Trening porównuje te próbki z polityką o `σ = 1` — stosunek
prawdopodobieństw jest astronomiczny, obcinane jest wszystko, a `target_kl`
przerywa aktualizację na starcie. Od iteracji 2 rollouty pochodzą z `σ = 1`
i KL wraca do rzędu setnych. **Reguła:** przy wznawianiu modelu z eksplodowanym
`σ` pod `log_std_max` jednorazowy skok KL jest oczekiwany; sprawdzać, że wraca
do normy w następnym zapisie. Gdyby nie wrócił, polityka nie aktualizowałaby się
wcale, a trening wyglądałby z zewnątrz na zdrowy.

**Do obserwacji, bez interpretacji:** `rollout/ep_rew_mean` w pierwszych czterech
iteracjach (~65k kroków) spada `0.564 → 0.390 → 0.302 → 0.166`. Nakładają się
tu cztery zmiany naraz — zacieśniany limit, spadek `σ` z ~15 do 1, horyzont
1000 → 2000 i obecny szum zamiast szumu z warunków `depress_ft` — a nagroda
w tym środowisku słabo mówi o skuteczności (6.30). Ocena dopiero przy
checkpoincie +1.5M przez `compare_models`.

#### Powtórka Kroku 1 na naprawionym limicie

Pary n = 40, h = 1500, domyślny protokół `compare_models` (chwytak zamknięty,
bramka palcowa). Przebieg bez limitu powtórzony w tej samej sesji dał **identyczne
liczby** jak w 6.31 (27/40 i 6/40, kolizje 16.2 i 8.5) — bez limitu hak jest
nieaktywny, ziarna deterministyczne, więc to potwierdzenie powtarzalności po naprawie.

Kolumna „6.31" to pomiar przy **błędnych limitach 1.3963 / 1.2218 i nieszczelnej
saturacji** — skażony podwójnie.

| model | pomiar | bez limitu | 6.31 | **naprawiony 0.8727** |
|---|---|---|---|---|
| baza 15.91M | sukcesy | 27/40 | 1/40 | **0/40** |
| | wciśnięte/ep | 4.05 | 1.18 | **0.075** |
| | kroki kolizji/ep | 16.2 | 75.5 | **41.5** |
| | misuse/ep | 0.23 | 13.2 | **14.93** |
| `hwlim` 18.91M | sukcesy | 6/40 | 5/40 | **0/40** |
| | wciśnięte/ep | 1.68 | 1.58 | **0.225** |
| | kroki kolizji/ep | 8.5 | 2.8 | **60.3** |
| | misuse/ep | 0.05 | 0.1 | **1.35** |

| porównanie (pary) | tylko pierwszy | tylko drugi | p |
|---|---|---|---|
| baza: bez limitu vs limit | 27 | 0 | **1.5e-08** |
| `hwlim`: bez limitu vs limit | 6 | 0 | **0.031** |
| pod limitem: baza vs `hwlim` | 0 | 0 | 1 |

**Wnioski.**

1. **Pod prawdziwym limitem firmware żaden istniejący model nie wykonuje zadania**
   — 0/40 dla obu.
2. **SPROSTOWANIE wniosku z 6.31** („`hwlim` jest jedynym modelem, którym wolno ruszyć
   robota", 2.8 kroku kolizji/ep). Była to wyłącznie cecha nieszczelnego limitu.
   Pod naprawionym limitem `hwlim` zderza się z tablicą **częściej niż baza**
   (60.3 wobec 41.5 kroku/ep). **Na dziś żaden model nie nadaje się na sprzęt.**
3. Skażone pomiary były optymistyczne **w skuteczności i w bezpieczeństwie**,
   zgodnie z kierunkiem błędu przewidzianym w ostrzeżeniu w 6.31.
4. `hwlim` trenowano w obwiedni, w której nadgarstek mógł kręcić się do ~2.9× szybciej.
   Pod realnym limitem jego wyuczone ruchy się nie mieszczą. **Model trenowany
   w luźniejszej obwiedni nie przenosi się nawet do nieco ciaśniejszej** — to
   bezpośredni argument, że trening musi odbywać się w obwiedni prawdziwej, czyli
   za biegiem `hwlim_curr`.

Artefakty (trwałe, poza katalogiem roboczym): `logs/board_6dof/eval_2026-09-13/`
— `fix_nolim1500.csv`, `fix_hw1500.csv`, `krok1_pairs.py`.

#### `depress_ft` w punktach kurikulum (naprawiony limit)

Protokół etapu 1 (chwytak swobodny, bez bramki), `random_order`, h = 2000,
20 epizodów na punkt:

| limit [rad/s] | sukcesy | wciśnięte/ep |
|---|---|---|
| brak | 15/20 | 4.60 |
| **2.0 (start kurikulum)** | **2/20** | **1.85** |
| 1.5 | 0/20 | 0.70 |
| 1.2 | 0/20 | 0.50 |
| 0.8727 (cel) | 0/20 | 0.15 |

**Założenie startu kurikulum jest słabsze, niż przyjęto.** Start 2.0 miał być
punktem, w którym `depress_ft` zachowuje umiejętność i uczy się wykonywać ją
wolniej. Pod naprawionym limitem już przy 2.0 skuteczność spada 15 → 2/20,
a wciśnięcia 4.60 → 1.85; sukcesy znikają przed 1.5.

**Błąd kolejności.** Bieg uruchomiono po teście wykonalności docisku przy 0.8727,
a przed tą krzywą, z uzasadnieniem, że służy już tylko dostrojeniu harmonogramu.
Krzywa pokazuje, że wybór punktu startowego miał znaczenie. To naruszenie reguły
z 6.31 w wersji słabszej — model przepuszczono przez punkt docelowy obwiedni,
ale nie przez punkt startowy.

**Konsekwencja dla `hwlim_curr`.** Kurikulum startuje w obszarze już zdegradowanym,
a limit 1.5 (0/20 dla `depress_ft`) przypada po ~1.11M kroków. Nie przesądza to
wyniku — PPO uczy się w nowej obwiedni, a krzywa dotyczy zamrożonej polityki —
ale checkpoint +1.5M wypada przy limicie ~1.32, gdzie model startowy nie ma już
sukcesów. **Kryterium na tym checkpoincie:** wciśnięcia/ep przy bieżącym limicie
wyraźnie powyżej `depress_ft` przy 1.2–1.5 (0.50–0.70).

#### Rozkład prędkości stawów `depress_ft` bez limitu — wybór punktu startowego

Protokół etapu 1, h = 1000, 10 epizodów, 8 507 kroków, `|qvel|` z obserwacji:

| staw | mediana | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| j1 | 0.099 | 0.320 | 0.441 | 0.676 | 1.345 |
| j2 | 0.192 | 0.790 | 1.057 | 1.644 | 3.387 |
| j3 | 0.558 | 1.586 | 2.031 | 3.228 | 6.765 |
| j4 | 0.795 | 2.188 | 2.552 | 3.963 | 17.99 |
| j5 | 0.313 | 1.714 | 2.168 | 2.835 | 4.255 |
| j6 | 1.244 | 2.473 | 2.893 | 4.453 | 16.75 |

Odsetek kroków, w których limit obciąłby ruch:

| limit [rad/s] | ≥ 1 staw | j3 | j4 | j5 | j6 |
|---|---|---|---|---|---|
| 0.8727 | 87.1 % | 33.1 | 46.4 | 27.3 | 64.3 |
| 1.2 | 77.2 % | 19.7 | 33.4 | 18.7 | 51.7 |
| 1.5 | 63.6 % | 11.6 | 24.6 | 12.9 | 40.4 |
| **2.0** | **37.3 %** | 5.2 | 13.5 | 6.4 | 21.5 |
| 2.5 | 16.2 % | 2.6 | 5.4 | 2.6 | 9.5 |
| **3.0** | **6.9 %** | 1.4 | 2.5 | 0.7 | 4.3 |
| **4.0** | **2.2 %** | 0.5 | 1.0 | 0.1 | 1.4 |
| 5.0 | 0.9 % | 0.1 | 0.6 | 0.0 | 0.7 |

**Wniosek.** Limit 2.0 obcina 37 % kroków, co tłumaczy spadek 15 → 2/20 w punkcie
startowym kurikulum. Obcięcie ok. jednej trzeciej kroków wystarcza, żeby rozbić
sekwencję pięciu przycisków. Punkt startowy zostawiający zachowanie modelu prawie
nietknięte leży dopiero przy **3.0–4.0** (6.9 % i 2.2 % kroków). Oba punkty są
weryfikowane bezpośrednio na `depress_ft` przed ewentualnym nowym biegiem —
rozkład prędkości nie mówi, ile z tych obciętych kroków jest krytycznych.

**Weryfikacja kandydatów na start** (`depress_ft`, protokół etapu 1, h = 2000,
20 epizodów, limit egzekwowany po całkowaniu):

| limit [rad/s] | sukcesy | wciśnięte/ep | kroki sukcesu |
|---|---|---|---|
| brak | 15/20 | 4.60 | — |
| **4.0** | **14/20** | **4.30** | 845 |
| **3.0** | **12/20** | **4.15** | 864 |
| 2.0 (start `hwlim_curr`) | 2/20 | 1.85 | — |
| 1.5 | 0/20 | 0.70 | — |

**Urwisko leży między 3.0 a 2.0.** Przy 3.0 model obcina 6.9 % kroków i traci
3 z 15 sukcesów, przy 2.0 obcina 37 % i traci 13. Bieg `hwlim_curr` wystartował
tuż za urwiskiem. Czy mimo to dostosowuje się do obwiedni, rozstrzyga parowane
porównanie jego checkpointu z `depress_ft` przy tym samym limicie (w toku).

CSV: `logs/board_6dof/eval_2026-09-13/depress_start_candidates.csv`.

#### Czy `hwlim_curr` dostosowuje się do obwiedni — checkpoint 14.53M

Checkpoint `ppo_board_14525856_steps.zip` (+300k, frac 0.12) kontra `depress_ft`
przy **tym samym** limicie 1.8647 (wartość kurikulum w chwili zapisu), protokół
etapu 1, h = 2000, 20 epizodów na tych samych ziarnach:

| model | sukcesy | wciśnięte/ep |
|---|---|---|
| `depress_ft` | 0/20 | 1.30 |
| checkpoint 14.53M | 0/20 | 1.45 |

Pary po ziarnie: checkpoint lepszy 6, gorszy 5, remis 9, **test znaków p = 1**.

**Wynik niekonkluzywny, ale z jasną przyczyną.** Po 300k kroków brak poprawy
względem modelu startowego, ale też brak degradacji. 300k to wcześnie, lecz
istotniejsze jest to, że przy obecnym limicie model startowy ma **0/20 sukcesów**,
a epizody treningowe dobijają do horyzontu (`ep_len_mean` ~1890/2000). PPO nie
dostaje sygnału ukończenia, z którego mógłby się uczyć — to ten sam problem
odkrycia, który zabił scratch `hwlim_s1`, tylko przesunięty.

**Rekomendacja:** zatrzymać `hwlim_curr` i wystartować od limitu 3.0, gdzie
`depress_ft` kończy 12/20. Zacieśnianie wydłużone, bo urwisko 3.0 → 2.0 zajmuje
prawie połowę zakresu. Konfiguracja przygotowana, niezaakceptowana:
`training/configs/ppo_board_6dof_hwlim_curr3.yaml`.

CSV: `logs/board_6dof/eval_2026-09-13/curr_ckpt14525856_vs_depress_L1.8647.csv`.

#### Restart od limitu 3.0 — bieg `hwlim_curr3`

**`hwlim_curr` zatrzymany 2026-09-13 o 21:39** na
checkpoincie `ppo_board_14725856_steps.zip` (+500k kroków, limit ≈ 1.77).
Checkpointy zostają w `logs/board_6dof/hwlim_curr/` jako materiał porównawczy.

**`hwlim_curr3` uruchomiony o 21:39:** `ppo_board_6dof_hwlim_curr3.yaml`,
`--resume logs/board_6dof/depress_ft/ppo_board_final.zip` (14.23M),
`--timesteps 5000000`, katalog `logs/board_6dof/hwlim_curr3`, TensorBoard
`ppo_board_6dof_hwlim_curr3_0`. Względem `hwlim_curr` zmienione tylko: start
3.0 zamiast 2.0, zacieśnianie 4M zamiast 2.5M, budżet 5M zamiast 4M.

Limit w funkcji kroku: `limit = 3.0 − frac · 2.1273`, `frac = (krok − 14 225 856) / 4 000 000`.

**Plan punktów kontrolnych** (parowanie z `depress_ft` przy tym samym limicie):

| checkpoint | kroki dodatkowe | limit | `depress_ft` w pobliżu |
|---|---|---|---|
| 15 225 856 | +1.0M | ≈ 2.47 | między 3.0 (12/20) a 2.0 (2/20) |
| 16 125 856 | +1.9M | ≈ 1.99 | 2.0: 2/20, 1.85 wciśnięcia |

Kryterium: wciśnięcia/ep wyraźnie powyżej `depress_ft` przy tym samym limicie,
szczególnie w drugim punkcie. Brak różnicy przy ~2.0 oznacza, że samo kurikulum
nie wystarcza.

**Weryfikacja startu `hwlim_curr3`** (TensorBoard):

| krok | limit (log) | limit (wzór) | `approx_kl` | `clip_fraction` | `ep_len_mean` | `ep_rew_mean` |
|---|---|---|---|---|---|---|
| 14 242 240 | 2.991 | 2.9913 | — | — | 868 | 0.652 |
| 14 258 624 | 2.983 | 2.9826 | **1035** | 1.000 | 926 | 0.628 |
| 14 275 008 | 2.974 | 2.9739 | 0.022 | 0.195 | 983 | 0.565 |
| 14 291 392 | 2.965 | 2.9652 | 0.017 | 0.147 | 1043 | 0.510 |

Kurikulum zgodne ze wzorem, `train/std = 1.0`, skok KL w iteracji 1 wrócił do normy —
**drugie potwierdzenie** reguły o artefakcie clampa przy wznowieniu modelu
z eksplodowanym `σ`.

**Sygnał ukończenia jest.** `ep_len_mean` 868–1043 wobec ~1890 w `hwlim_curr`
przy limicie ~1.9 — epizody się kończą, co było celem restartu.

**Do obserwacji:** w pierwszych czterech zapisach (~65k kroków) długość epizodu
rośnie, a nagroda spada. Nakłada się na to spadek `σ` z ~15 do 1 i początek
zacieśniania. Trend sprawdzany przy +500k, ocena skuteczności na punktach kontrolnych.

**Trend przy +500k** (TensorBoard, `approx_kl` z najbliższego zapisu treningu):

| krok | limit | `ep_len_mean` | `ep_rew_mean` | `approx_kl` |
|---|---|---|---|---|
| 14 242 240 | 2.991 | 868 | 0.652 | — |
| 14 340 544 | 2.939 | 1201 | 0.423 | 0.019 |
| 14 438 848 | 2.887 | 1286 | 0.271 | 0.015 |
| 14 537 152 | 2.834 | 1308 | 0.281 | 0.027 |
| 14 635 456 | 2.782 | 1337 | 0.257 | 0.016 |
| 14 733 760 | 2.730 | **1324** | **0.358** | 0.022 |

Po spadku w pierwszych ~150k kroków (`σ` z ~15 do 1, początek zacieśniania)
**długość epizodu stabilizuje się w okolicy 1310–1340** i nie dąży do horyzontu
2000, choć limit schodzi z 2.99 do 2.73. Nagroda odbija z ~0.25 do 0.36. Sygnał
ukończenia jest zachowany — zachowanie jakościowo odmienne od `hwlim_curr`
(~1890 przy limicie ~1.9). Bieg kontynuowany do punktu kontrolnego.

**Punkt kontrolny 1 — checkpoint 15 225 856** (+1.0M, frac 0.25, limit 2.4682),
`ckpt_eval.py`, protokół etapu 1, h = 2000, 20 epizodów na tych samych ziarnach:

| model | sukcesy | wciśnięte/ep |
|---|---|---|
| `depress_ft` (model startowy) | 8/20 | 3.60 |
| checkpoint +1.0M | 5/20 | 2.75 |

| pary | lepszy checkpoint | lepszy `depress_ft` | remis | p |
|---|---|---|---|---|
| wciśnięcia | 4 | 9 | 7 | 0.267 (test znaków) |
| sukcesy | 3 | 6 | — | 0.508 |

Średnia różnica wciśnięć (checkpoint − start): **−0.85**.

**Kryterium niespełnione, kierunek przeciwny.** Różnica nie jest istotna, ale po
milionie kroków kurikulum polityka radzi sobie z obwiednią **gorzej** niż sam model
startowy przy tym samym limicie — mimo że przy 2.47 sygnał ukończenia był
(`depress_ft` 8/20, `ep_len_mean` ~1320).

**Decyzja: bieg dochodzi do punktu 2 bez zmian.** Kryterium zapisano przed
obejrzeniem danych i wskazano w nim punkt 2 (limit ≈1.99) jako rozstrzygający.
Przerwanie na podstawie nieistotnego wyniku w łagodniejszym punkcie byłoby
przesuwaniem bramki. Jeśli punkt 2 również nie pokaże zysku, kurikulum po limicie
uznaje się za obalone.

**Sprostowanie argumentu o `output_max`.** Wcześniej przestrzeń akcji obrotu opisano
jako jedenastokrotnie szerszą od wykonalnej (komenda OSC 10 rad/s wobec limitu
stawu 0.87). Porównanie było błędne — sam OSC realizuje tylko ułamek komendy.
Zmierzona utrata na obrocie to ~2× (0.48–0.59). Zmiana `output_max` pozostaje
kandydatem strukturalnym, ale z tym słabszym uzasadnieniem.

CSV: `logs/board_6dof/eval_2026-09-13/ckpt_15225856_vs_depress_L2.4682.csv`.

**Awaria: OOM 2026-09-14 o 02:10 — bieg `hwlim_curr3` padł.** Jądro zabiło
proces `python` (anon-rss 2.3 GB) oraz procesy Webex (`CiscoCollabHost`).
Bezpośredni wyzwalacz: automatyczna ocena punktu 2 wystartowała o 02:08 obok
treningu z 8 środowiskami i otwartego Webeksa, na 15 GB RAM.

Traceback z logu treningu precyzuje mechanizm: `EOFError` w
`SubprocVecEnv.step_wait`, wywołanym z `EvalCallback → evaluate_policy`. Padł
**podproces środowiska ewaluacyjnego** wbudowanej ewaluacji, a proces główny
treningu zginął na zerwanym potoku. Wbudowana ewaluacja (20 epizodów po 2000
kroków przy limicie 0.8727) startuje co 100k kroków — **na tej samej granicy co zapis
checkpointu**. Czuwanie wyzwalane pojawieniem się checkpointu uruchomiło więc
zewnętrzną ocenę dokładnie w chwili, gdy trening sam ewaluował, co zmaksymalizowało
nakładanie się pamięci. To błąd projektowy czuwania, nie przypadek.

Ostatni checkpoint `ppo_board_16125856_steps.zip` zapisał się o 02:07 i jest
**dokładnie punktem kontrolnym 2** (limit ≈ 1.99). Strata to ok. 3 minuty treningu,
a rozstrzygającą ocenę można wykonać bez wznawiania biegu.

**Pułapka przy wznowieniu.** `JointVelLimitCurriculum` liczy `frac` od `num_timesteps`
w chwili startu biegu. `--resume` z checkpointu 16.13M rozpocząłby zacieśnianie
**od nowa od 3.0** zamiast od ~1.99. Wznowienie w środku kurikulum wymaga parametru
początku, np. stałego kroku odniesienia 14 225 856 w konfiguracji.

**Reguła:** nie uruchamiać ocen równolegle z treningiem 8-środowiskowym przy otwartych
aplikacjach typu Webex — 15 GB RAM nie ma zapasu na trzeci proces z modelem PPO. Zewnętrzne oceny wyzwalać
**z opóźnieniem po checkpoincie** (po zakończeniu wbudowanej ewaluacji) albo po
zatrzymaniu treningu — nigdy na samo pojawienie się pliku checkpointu.

**Log jądra (02:09–02:15).** Presja pamięci zaczęła się o 02:09:13, minutę po starcie
zewnętrznej oceny; zabójcę OOM wywołał proces `python`. Jądro najpierw ubijało
procesy Webex (`oom_score_adj` 200), o 02:10:24 zabiło `python` z 2.3 GB i trening
padł. **W oknie 02:11–02:15 brak wpisów OOM jądra** — ponowne ubicie oceny o 02:11
było decyzją harnessu przy odzyskującym pamięć systemie, nie skutkiem tej oceny.

**Przyczyna głębsza: cykl referencji w haku limitu.** Proces z 2.3 GB wymagał
wyjaśnienia, więc sprawdzono narastanie pamięci: 120 twardych resetów
`ButtonBoardEnv` (horyzont 50), trzy warianty, każdy w osobnym procesie.

| wariant | RSS po resetach 0 / 20 / 40 / 60 / 80 / 100 / 119 [MB] |
|---|---|
| hak jak wyżej | 524 / **957** / 893 / 628 / **951** / 860 / 790 |
| bez haka | 508 / 578 / 578 / 519 / 533 / 566 / 578 |
| hak + `gc.collect()` po resecie | 504 / 563 / 563 / 563 / 593 / 563 / 563 |

Z hakiem pamięć rośnie **piłokształtnie** i opada dopiero przy przebiegu cyklicznego
GC; bez haka i z wymuszonym GC jest płaska. Mechanizm: domknięcie przechwytywało
metodę związaną `sim.step2`, która trzyma obiekt `sim`, a `sim` trzymał domknięcie —
cykl. Po twardym resecie stary `MjSim` z pamięcią C MuJoCo nie zwalniał się przez
licznik referencji, tylko czekał na cykliczny GC, wyzwalany liczbą alokacji obiektów
Pythona, a nie rozmiarem pamięci C.

W treningu 9 procesów środowisk przez 4.5 h mogło akumulować martwe symulacje między
przebiegami GC. Jest to **prawdopodobne** pochodzenie 2.3 GB w zabitym procesie,
ale nie zmierzone w warunkach treningu. Zgodne z tym jest, że ten sam układ procesów
przez 2 h biegu `hwlim_curr` nie padł.

**Naprawa:** hak woła implementację z klasy na bieżącym `self.sim`
(`type(sim).step2(self.sim)`) i nie przechwytuje metody związanej — domknięcie nie ma
referencji do obiektu `sim`, więc stary `MjSim` zwalnia się od razu po resecie.
**Weryfikacja poprawki (2026-09-14).**

*Dokładność limitu.* Test z 6.32 powtórzony po poprawce — wszystkie sześć warunków OK,
wartości **identyczne co do cyfry** z weryfikacją sprzed poprawki (np. per staw przy
0.8727: `[0.5818, 0.8727, 0.8727, 0.8727, 0.8727, 0.8727]`). Zmiana sposobu wywołania
całkowania nie zmieniła fizyki.

*Pamięć.* 600 twardych resetów, RSS co 50 resetów, nachylenie MNK od resetu 100,
każdy wariant w osobnym procesie:

| wariant | zakres RSS | max | nachylenie |
|---|---|---|---|
| bez haka (odniesienie) | 548–633 MB | 633 MB | +5.8 MB/100 resetów |
| **hak po poprawce** | **533–577 MB** | **577 MB** | **−1.6 MB/100 resetów** |
| hak przed poprawką (120 resetów) | 524–957 MB | 957 MB | piłokształtnie |

Poprawiony hak mieści się w pasmie szumu wariantu bez haka, a nachylenie jest zerowe
w granicach szumu. Wynik krótkiego testu po poprawce (+49 MB/100 przy 120 resetach)
był szumem alokatora — rozstrzygnął go dopiero dłuższy przebieg z odniesieniem.
**Wyciek usunięty, poprawka zweryfikowana w obu wymiarach.** Zapas pamięci na długi
trening wraca, ale reguła o nieuruchamianiu ocen obok treningu przy otwartym Webeksie
obowiązuje nadal.

**Reguła:** każdy hak instalowany na obiekcie symulacji trzeba sprawdzić testem pamięci
przez wiele twardych resetów, nie tylko testem poprawności fizyki.

#### Punkt kontrolny 2 — checkpoint 16 125 856 (limit 1.9895)

Oceniony po awarii OOM, na wolnej pamięci, bez wznawiania biegu — checkpoint zapisał
się o 02:07, trzy minuty przed awarią. Protokół etapu 1, h = 2000, 20 epizodów na tych
samych ziarnach:

| model | sukcesy | wciśnięte/ep |
|---|---|---|
| `depress_ft` (model startowy) | 0/20 | 1.15 |
| checkpoint +1.9M | 2/20 | **2.25** |

| pary | lepszy checkpoint | lepszy `depress_ft` | remis | p |
|---|---|---|---|---|
| wciśnięcia | **11** | **3** | 6 | **0.057** (test znaków) |
| sukcesy | 2 | 0 | — | 0.5 |

Średnia różnica wciśnięć (checkpoint − start): **+1.10**, czyli +96 %.

**Werdykt według kryterium zapisanego z góry: kurikulum NIE jest obalone.** Warunkiem
obalenia był „brak różnicy przy ~2.0". Różnica jest duża i spójna co do kierunku (11:3),
choć przy n = 20 statystycznie graniczna (p = 0.057, tuż powyżej 0.05).

**Obraz łączny z punktem 1** (limit 2.47: checkpoint gorszy, −0.85, pary 4:9). Polityka
oddała część skuteczności w luźnej obwiedni, a zyskała w ciasnej, poniżej urwiska, gdzie
model startowy traci sukcesy. To zamierzony efekt kurikulum — adaptacja do zacieśniającej
się obwiedni kosztem wydajności przy limicie, którego docelowo nie będzie.

**Potwierdzenie przed decyzją o wznowieniu:** ocena rozszerzona do n = 40 (ziarna 20–39).
Uwaga metodyczna: rozszerzenie próby po zobaczeniu granicznego wyniku to forma testowania
sekwencyjnego i lekko zawyża ryzyko wniosku fałszywie dodatniego. Wynikiem zarejestrowanym
z góry pozostaje n = 20, a n = 40 jest potwierdzeniem.

**Wznowienie wymaga poprawki kurikulum** — stałego kroku odniesienia `joint_vel_anneal_origin`
= 14 225 856, żeby limit kontynuował od ~1.99 zamiast startować od 3.0.

**Poprawka zrobiona.** `JointVelLimitCurriculum` przyjmuje `origin`; `_on_training_start`
ustawia `t0 = origin` i od razu nakłada limit właściwy dla **bieżącego** kroku, nadpisując
także wartość, z którą zbudowano środowiska treningowe. Bez `origin` zachowanie jest
identyczne jak dotąd. Test na `SubprocVecEnv` z odczytem z podprocesów:

| wariant | licznik startu | pierwszy limit | oczekiwany | odczyt z env |
|---|---|---|---|---|
| `origin` 10.0M, wznowienie przy 10.5M | 10 500 000 | 1.93635 | 1.93635 | = ostatni zapis |
| bez `origin` | 10 500 000 | 3.00000 | 3.00000 | = ostatni zapis |

Pierwsze uruchomienie testu zgłosiło błąd w obu wariantach mimo poprawnych liczb — test
porównywał limity zaokrąglone do 5 miejsc z pełną wartością z env przy tolerancji 1e-9.
Poprawiono porównanie i powtórzono test do czystego przejścia.

**Konfiguracja wznowienia przygotowana, nieuruchomiona:**
`training/configs/ppo_board_6dof_hwlim_curr3b.yaml` — harmonogram identyczny z `hwlim_curr3`,
`joint_vel_anneal_origin: 14225856`, nowy katalog `logs/board_6dof/hwlim_curr3b`, start z
`ppo_board_16125856_steps.zip`, `--timesteps 3100000`. Kontrola: limit przy wznowieniu
**1.9895** (frac 0.4750) — ciągły względem checkpointu. Checkpoint ma już `σ = 1.0`,
więc skok KL w pierwszej iteracji nie jest oczekiwany.

#### Potwierdzenie punktu 2 na n = 40 — przewaga NIE powtórzyła się

Ziarna 20–39 ocenione osobno i połączone z 0–19, ten sam protokół
(`ckpt_eval.py 16125856 20 20`, `combine_pairs.py`):

| próba | `depress_ft` sukcesy | wciśnięte/ep | checkpoint sukcesy | wciśnięte/ep | pary wciśnięć lepszy:gorszy (remis) | p | różnica |
|---|---|---|---|---|---|---|---|
| ziarna 0–19 (zarejestrowane) | 0/20 | 1.15 | 2/20 | 2.25 | 11:3 (6) | 0.057 | +1.10 |
| ziarna 20–39 | 3/20 | 1.95 | 1/20 | 1.85 | 8:8 (4) | 1.0 | −0.10 |
| **łącznie 0–39** | **3/40** | **1.55** | **3/40** | **2.05** | **19:11 (10)** | **0.20** | **+0.50** |

Sukcesy łącznie w parach: 3:3, p = 1.

**Wynik z n = 20 był w dużej części szumem.** Druga połowa ziaren nie pokazuje przewagi,
a łączny efekt maleje do +0.50 wciśnięcia bez istotności i bez różnicy w sukcesach.
To dokładnie ryzyko zaznaczone przy rozszerzaniu próby: graniczny wynik na małej próbie
regresuje do średniej.

**Werdykt skorygowany:** brak wiarygodnych dowodów, że kurikulum poprawiło zachowanie
w obwiedni firmware. Punkt 1 ujemnie (−0.85), punkt 2 przy n = 40 nieistotnie dodatnio
(+0.50, p = 0.20), sukcesy bez zmian. **Konfiguracja `hwlim_curr3b` nie zostaje uruchomiona.**

**Stan łączny prób treningu w obwiedni 0.8727:**

| próba | wynik w obwiedni firmware |
|---|---|
| fine-tune bazy `hwlim` | 0/40 (naprawiony limit) |
| scratch `hwlim_s1` | 0/40, zero wciśnięć (trenowany na nieszczelnym limicie) |
| kurikulum od 2.0 `hwlim_curr` | brak adaptacji przy 1.86 (p = 1) |
| kurikulum od 3.0 `hwlim_curr3` | brak wiarygodnej przewagi przy 1.99 (n = 40, p = 0.20) |

Żaden wariant nie przywrócił skuteczności w prawdziwej obwiedni w testowanych budżetach.

**Uwaga metodyczna do pracy:** wynik graniczny przy n = 20 omal nie uzasadnił wznowienia
kilkugodzinnego biegu. Decyzje kosztowne w obliczeniach wymagają potwierdzenia na
niezależnych ziarnach, a nie tylko spełnienia kryterium na próbie wstępnej.

CSV: `logs/board_6dof/eval_2026-09-13/ckpt_16125856_vs_depress_L1.9895_s20-39.csv`.

CSV: `logs/board_6dof/eval_2026-09-13/ckpt_16125856_vs_depress_L1.9895.csv`.

### 6.33. Siła nacisku E-stopu i nagrania z German Open (2026-09-14)

#### Model przycisku jest za MIĘKKI

Źródło: typowa specyfikacja przemysłowych E-stopów 22/30 mm —
**siła aktywacji 20–40 N** (celowo wymaga zdecydowanego nacisku). Jednostką jest N,
nie N/m; N·m dotyczy tylko momentu dokręcania pierścienia montażowego. To wartość
katalogowa, nie pomiar tablicy z uczelni — pomiar nadal jest w prośbie do laboratorium.

Nasz model (`my_assets/objects/xml/button.xml`):

| parametr | wartość |
|---|---|
| sztywność sprężyny | 500 N/m |
| próg aktywacji | 12 mm |
| tarcie (`frictionloss`) | 1 N |
| **siła do aktywacji** | **≈ 7 N** (500 × 0.012 + 1) |

Prawdziwy przycisk wymaga **3–6× większej siły**. Liniowa sprężyna o sile 20–40 N przy
12 mm to 1 700–3 300 N/m. Zastrzeżenie: E-stop jest mechanizmem zatrzaskowym (siła rośnie
do progu, potem przeskok), więc liniowa sprężyna jest przybliżeniem. Skok do zatrzasku
pozostaje nieznany.

**Sprostowanie wcześniejszego przypuszczenia.** W 6.31–6.32 rozważano, że model grzybka
może być za twardy i utrudniać docisk pod limitem. Dane wskazują odwrotnie: na sprzęcie
docisk będzie **trudniejszy** niż w symulacji. Model przycisku trzeba poprawić przed
dalszym treningiem pod sprzęt — ta sama klasa poprawki wierności co geometria (6.24)
i limity prędkości (6.31).

#### German Open — wciskanie zamkniętym chwytakiem

Zrzuty z nagrania zespołu (Instagram `rrl_alert`, #germanopen #fhaachen), 2026-09-14:

- robot wciska przyciski **zamkniętym chwytakiem**, tak jak w naszym sformułowaniu zadania;
  taśma między otwartymi palcami dotyczyła innego turnieju (Korea);
- tablica na ścianie ze skośnymi gniazdami, jak w naszym modelu;
- ramię dojeżdża **wyciągnięte poziomo, z chwytakiem skierowanym w tablicę**.

**Różnica wobec naszego zadania: poza startowa.** Nasz reset ustawia chwytak pionowo w dół
(`eef_align` 89.9°, 6.27), więc przed pierwszym dociskiem polityka obraca nadgarstek o ~65–90°.
Obrót to jedyna oś realnie dławiona limitem firmware (0.48–0.59×, 6.32). Z nagrania nie da się
zmierzyć orientacji, ale kierunek różnicy jest wyraźny. To kandydat do sprawdzenia, nie wniosek.

#### Otwarte pytania przed dalszym treningiem

1. Czy obecny układ (OSC `kp` 150, `output_max` 0.05) w ogóle wytwarza 20–40 N na grzybku —
   pomiar maksymalnej siły kontaktu na grzybku. **Uwaga:** czujnik `button_force` z `button.xml`
   **nie trafia do skompilowanego modelu** — środowisko ma tylko `gripper0_right_force_ee`
   i `gripper0_right_torque_ee`. Siła jest więc liczona z kontaktów MuJoCo (`mj_contactForce`,
   suma sił normalnych na geometrii `btn{i}_cap_collision`). Pierwsza wersja pomiaru zakończyła
   się na braku czujnika, zanim cokolwiek zmierzyła.
2. Jak zmienia się skuteczność znanego modelu po podniesieniu sztywności do 1 700–3 300 N/m
   (reguła z 6.31: najpierw znany model przez nową obwiednię).
3. Czy poza startowa z chwytakiem skierowanym w tablicę zmniejsza koszt limitu prędkości.

#### Pytanie 1 — siła nacisku w symulacji (pomiar z kontaktów)

`depress_ft`, protokół etapu 1, h = 1000, 10 epizodów. Siła normalna z `mj_contactForce`,
sumowana po kontaktach na `btn{i}_cap_collision` z pominięciem innych geometrii przycisków:

| wielkość | bez limitu | limit 0.8727 |
|---|---|---|
| siła w chwili aktywacji 12 mm, mediana | 6.4 N (10/10 epizodów) | 4.9 N (3/10) |
| siła przy ograniczniku skoku 15 mm | 7.2 N (1/10) | nieosiągnięty |
| max siła na grzybku w epizodzie, mediana | 24.1 N | 18.0 N |
| max siła, p90 / max | 42.8 / 43.3 N | 75.8 / 90.9 N |
| epizody z siłą ≥ 20 N / ≥ 40 N | 7/10 / 2/10 | 4/10 / 4/10 |
| czujnik chwytaka `force_ee`, mediana max | 148.8 N | 139.4 N |

**Metoda działa.** Siła aktywacji 6.4 N zgadza się z modelem (500 N/m × 12 mm = 6 N, plus tarcie).

**Siły 20–90 N to impulsy uderzeniowe, nie docisk.** Maksima występują, gdy grzybek nie
jest przy ograniczniku; ogranicznik osiągnięto raz, przy 7.2 N. Polityka dociska dokładnie
tyle, ile wymaga miękki model (~6 N), a duże wartości pochodzą ze zderzeń. Pod limitem
impulsy są silniejsze (p90 75.8 N) przy mniejszej liczbie aktywacji (3/10) — spowolniony
nadgarstek częściej uderza w grzybek, niż go przepycha. Czujnik chwytaka (140–150 N)
jest zdominowany przez bezwładność i nie mierzy nacisku na przycisk.

**Odpowiedź częściowa.** Symulacja potrafi chwilowo wytworzyć 20–40 N, ale nie wiadomo,
czy sterownik utrzyma taki nacisk quasi-statycznie przez 12 mm skoku. Rozstrzyga pytanie 2
— znany model na sztywniejszym przycisku. Do środowiska dodano parametr `button_stiffness`
(domyślnie `None`, bez zmiany zachowania).

**Parametr `button_stiffness`** (`ButtonBoardEnv`, `compare_models --button-stiffness`)
nadpisuje `jnt_stiffness` stawów `btn{i}_button_slide` po kompilacji modelu, przy każdej
inicjalizacji `sim` (twardy reset tworzy nowy obiekt). Weryfikacja fizycznego skutku —
siła sprężyny `qfrc_passive` przy 12 mm, na wszystkich pięciu przyciskach, po pierwszym
i drugim resecie:

| `button_stiffness` | sztywność w modelu | siła sprężyny @12 mm | oczekiwana |
|---|---|---|---|
| `None` | 500 N/m | −6.0 N | −6.0 N |
| 1700 | 1700 N/m | −20.4 N | −20.4 N |
| 3300 | 3300 N/m | −39.6 N | −39.6 N |

Tłumienie (20 N·s/m), tarcie (1 N), skok (15 mm) i próg (12 mm) bez zmian. Zmiana dotyczy
modelu obiektu zadania, nie robota ani chwytaka.

CSV: `logs/board_6dof/eval_2026-09-14/press_force.csv`.

#### Karta katalogowa E-stopu i pochodzenie progu 12 mm

Dane z karty katalogowej przycisku bezpieczeństwa (2026-09-14):

| parametr | karta | nasz model |
|---|---|---|
| skok roboczy (actuating stroke) | **5 mm** | 15 mm (`range`) |
| droga do rozwarcia styku (positive break travel) | 2 mm | — |
| rozwarcie styków NO / NC | 3 mm / 3 mm | — |
| typowa siła aktywacji | **25 N** | ≈ 6 N przy 12 mm |
| aktywacja | zatrzask przy pełnym skoku | próg 12 mm |
| trwałość mechaniczna | 100 000 cykli | — |

Typowy przemysłowy E-stop 22/30 mm ma skok 4–6 mm i blokuje się dopiero po wciśnięciu
do końca.

**Skąd 12 mm.** Z pierwszego commita projektu (2026-07-07): `button.xml` ze skokiem 15 mm
i sprężyną 500 N/m oraz `ACTIVATION_DEPTH = 0.012`. Wartości były arbitralnym wyborem
projektowym symulacji i nigdy nie zostały zmierzone. Rozdział 6.10 odnotował to w lipcu
(„12 mm było arbitralnym wyborem projektowym symulacji, realna głębokość aktywacji Omniboardu
nie jest znana"), a opis „6 N — realistyczna dla E-Stopa" w rozdz. 4.3 nie miał źródła.

**Model odbiega od karty w obu kierunkach:**

- **siła:** 6 N wobec 25 N — model ~4× za miękki;
- **głębokość:** 12 mm wobec 5 mm skoku — model wymaga 2.4× głębszego docisku,
  niż fizycznie istnieje.

Liniowa sprężyna dająca 25 N przy 5 mm to **5000 N/m** (przybliżenie, mechanizm jest
zatrzaskowy).

**Konsekwencja dla dotychczasowych wyników.** Część trudności 6DOF dotyczyła samego dojścia
do 12 mm: plateau 9–11 mm i brak gradientu między 7.6 a 12 mm (6.10, 6.16). Na prawdziwym
przycisku takiej głębokości nie ma. Zadanie w symulacji było **trudniejsze głębokościowo
i łatwiejsze siłowo** niż rzeczywistość.

**Konsekwencja dla pomiarów w toku.** Testy sztywności 1700 / 3300 N/m zachowują próg 12 mm,
więc nie odwzorowują karty. Pozostają informatywne dla pytania, czy sterownik utrzymuje nacisk
20–40 N. Wierny model to skok 5 mm, aktywacja przy ~5 mm (zatrzask; styk elektryczny
~2–3 mm), 25 N, k ≈ 5000 N/m. Zmiana skoku wymaga nadpisania `jnt_range` obok
`button_stiffness` i **unieważnia porównywalność** z dotychczasowymi wynikami — decyzja
do podjęcia.

#### Pytanie 2 — model na sprzęt na sztywnym przycisku (bez limitu prędkości)

`compare_models --button-stiffness`, `ft_sc1_v23_long`, protokół sprzętowy (chwytak zamknięty,
bramka palcowa), próg 12 mm, h = 1500, n = 20:

| sztywność | siła @12 mm | sukcesy | wciśnięcia łącznie (btn0–4) | kroki kolizji/ep |
|---|---|---|---|---|
| 500 N/m | 6 N | 13/20 | 81 | 3.0 |
| 1700 N/m | 20.4 N | **0/20** | 2 | 41.5 |
| 3300 N/m | 39.6 N | **0/20** | 0 | 2.5 |

**Sama siła przycisku, bez żadnego limitu prędkości, wyłącza model na sprzęt.** Przy sile
rzędu realnego E-stopu skuteczność spada z 13/20 do 0/20.

**Interpretacja ostrożna.** Polityka nauczyła się dociskać z siłą ~6 N, bo tyle wymagał
model (pomiar z kontaktów: aktywacja przy 6.4 N). Przy 1700 N/m napiera i uderza
(41.5 kroku kolizji), przy 3300 N/m praktycznie nie próbuje (2.5). Wynik pokazuje, że
**wytrenowana polityka nie dociska z siłą 20 N**, ale nie rozstrzyga, czy sterownik OSC
**potrafi** utrzymać taki nacisk. Rozstrzyga to test zdolności sterownika (w toku):
polityka doprowadza chwytak do kontaktu, potem stała komenda docisku i pomiar siły ustalonej.

Zastrzeżenie: próg 12 mm nie odpowiada karcie (skok 5 mm), więc 1700 N/m × 12 mm to
jednocześnie 20 N i 2.4× za głęboki docisk.

CSV: `logs/board_6dof/eval_2026-09-14/stiffness_base_k{500,1700,3300}.csv`.

#### Pytanie 2 — `depress_ft` na sztywnym przycisku i zdolność sterownika

`depress_ft`, protokół etapu 1, bez limitu prędkości, h = 2000, n = 20, próg 12 mm:

| sztywność | siła @12 mm | sukcesy | wciśnięte/ep | aktywacje | siła w chwili aktywacji (mediana) |
|---|---|---|---|---|---|
| 500 N/m | 6.0 N | 15/20 | 4.60 | 92 | 6.0 N |
| 1700 N/m | 20.4 N | 0/20 | 1.05 | **22** | **19.9 N** |
| 3300 N/m | 39.6 N | 0/20 | 0.05 | 1 | 39.5 N |

**Sterownik potrafi docisnąć ~20 N.** 22 aktywacje przy 1700 N/m, każda przy ~19.9 N,
oznaczają nacisk utrzymany przez całe 12 mm skoku — nie impulsy uderzeniowe. Przy 3300 N/m
jedna aktywacja przy 39.5 N: 40 N jest osiągalne, ale rzadko przy tej polityce. Wartość
z karty katalogowej (25 N) mieści się w przedziale wykazanym w symulacji.

**Test „stała komenda docisku po kontakcie" jest nieważny jako pomiar zdolności.**
`controller_force.py`: po kontakcie `depress_ft` przejmowała stała komenda wzdłuż osi
przycisku na prawie sztywnym grzybku (1e5 N/m). Mediana siły ustalonej wyszła 0.0 N,
bo kontakt był utrzymany tylko w 23–37 % z ostatnich 40 kroków (maksima 24–32 N) —
ślepe pchanie od przypadkowego punktu kontaktu zsuwa chwytak z wypukłego grzybka.
Wynik nie mówi nic o tym, ile sterownik utrzyma; rozstrzygający jest pomiar aktywacji wyżej.

**Werdykt.** Załamanie przy realnej sile to ograniczenie **wyuczonej polityki**, nie
sterownika. Obie polityki — etapu 1 i model na sprzęt — nauczyły się dociskać z siłą
~6 N, bo tyle wymagał arbitralny model przycisku. **Brak realistycznej siły przycisku
jest luką sim2real niezależną od limitów prędkości** i co najmniej równie istotną:
model na sprzęt traci całą skuteczność przy 20 N nawet bez limitu.

**Kierunek:** przycisk według karty (skok 5 mm, aktywacja ~5 mm, 25 N, k ≈ 5000 N/m),
dotrenowanie na nowym przycisku najpierw bez limitu prędkości (jedna zmienna naraz),
potem z limitem 0.8727. Zmiana modelu przycisku wymaga osobnej decyzji, bo
unieważnia porównywalność z dotychczasowymi wynikami.

CSV: `logs/board_6dof/eval_2026-09-14/stiffness_depress.csv`, `controller_force.csv`.

### 6.34. Nieliniowy model E-stopu i trening docelowy na sprzęt (2026-09-14)

**Model przycisku.** Liniowa sprężyna (500 N/m, ~6–7 N przy 12 mm) była błędem założeń:
przemysłowy E-stop nie jest sprężyną. Opór rośnie do punktu rozwarcia styków, potem
gwałtownie spada, a przycisk zatrzaskuje się przy pełnym skoku. Nowy parametr
`button_estop_profile` w `ButtonBoardEnv` (domyślnie `None` = stary model, bez zmian):

- siła przeciwna wciskaniu rośnie liniowo `f_start → f_peak` do `x_peak`,
- na odcinku `snap_width` spada do `f_after` i dalej jest stała,
- skok przycięty do `x_stroke`, przy `latch: true` grzybek zostaje na dole do resetu,
- siła wchodzi przez `qfrc_applied` w każdym substepie (hak `_limited_integration`),
  `jnt_stiffness = 0`.

Weryfikacja (`logs/board_6dof/eval_2026-09-14/verify_estop.py`, wszystko OK):
siła na 5 przyciskach równa krzywej, rampa quasi-statyczna 33 N zatrzymuje grzybek na
3.18 mm (oczekiwane 3.15), 37 N zatrzaskuje, reset zwalnia zatrzask, 40 twardych resetów
bez wzrostu RSS. Regresja: bez profilu obs i reward identyczne bit w bit (domyślny, k=1700,
limit 0.8727).

Dwie pułapki, obie w kodzie:

- miękki limit stawu MuJoCo (solref 0.02) przy grzybku 36 g ustępował 8 mm pod siłą
  wstępną 15 N — oba końce skoku są ogranicznikiem kinematycznym (clamp po całkowaniu);
- przy dt = 2 ms obciążenie przyłożone skokowo przeskakuje szczyt ~9 % wcześniej
  (32 N wobec 35 N) przez przeregulowanie dyskretne. Przy narastającym docisku próg jest dokładny.

**Trening docelowy.** Zamiast dalszych testów diagnostycznych —
trening w pełnej obwiedni sprzętowej: limit 0.8727 rad/s na wszystkich stawach oraz
E-stop **10 → 20 N do 4 mm, spadek do 5 N, skok 6 mm, zatrzask = wciśnięcie**
(`activation_depth` 5.9 mm). 20 N z zapasem względem przycisku z zawodów. Poza startowa
poza zakresem — obsługuje ją supervisor. Uzasadnienie wykonalności: zespół wcisnął
przyciski na RoboCup na prawdziwym Gen3 zamkniętym chwytakiem.

Config `training/configs/ppo_board_6dof_estop_hw.yaml`: fine-tune z `ft_sc1_v23_long`
(15.91M), +6M, h = 1500, reward v2.3 bez zmian, `log_std_max` 0.0, `target_kl` 0.03.
Katalog `logs/board_6dof/estop_hw`. Start 2026-09-14, 285 fps (~6 h).

**Sprostowanie — fałszywe „zapominanie” (pułapka pomiarowa).** Pierwszy bieg (`estop_hw`)
pokazał `ep_len_mean` 752 → 1500 i `ep_rew_mean` 0.38 → −1.66 w ~180k kroków. Zinterpretowano to jako
zapominanie i uruchomiono trening ponownie z ostrożniejszą aktualizacją (`estop_hw_lr`: lr 3e-5, target_kl 0.01).
Przesłanka była fałszywa. SB3 zapisuje w zipie modelu bufor 100 ostatnich epizodów
(`ep_info_buffer`), a `--resume` go wczytuje — pierwsze wartości `rollout/ep_*` pochodziły
ze starego treningu `ft_sc1_v23_long` (śr. długość 678 przy h = 1000, reward 0.62) i były
stopniowo wypierane przez nowe epizody. Dowody:

- długości epizodów identyczne co do kroku w obu biegach przez 4 iteracje mimo innego lr
  (rewardy już różne) — nowe epizody miały zawsze 1500 kroków, niezależnie od polityki;
- ewaluacja przy 16.01M: 20/20 epizodów do horyzontu;
- env odtworzony jak w treningu: model bazowy 0/5 wciśnięć w epizodach 0–2
  (deterministycznie i stochastycznie), `eval_2026-09-14/first_episode_check.py`.

**Model bazowy w obwiedni docelowej (limit 0.8727 + E-stop 20 N) startuje od zera** — nie
było czego zapominać. Po wznowieniu `rollout/ep_*` są niewiarygodne przez ~100 × horyzont
kroków; wnioski tylko z `eval/`.

**Etap A — curriculum siły E-stopu, bez limitu prędkości** (decyzja 2026-09-14). Obie ściany
są osobno śmiertelne dla modelu bazowego (sam limit: 27/40 → 0/40; sam przycisk 20 N:
13/20 → 0/20), więc jedna zmienna naraz. `EstopForceCurriculum` w `train_board.py`
interpoluje liczbowe klucze profilu od `button_estop_profile_start` do `button_estop_profile`
przez `button_estop_anneal_steps`, z odczytem zwrotnym ze wszystkich podprocesów (metody
`set_estop_profile`/`get_estop_profile` na wrapperze, nie `set_attr`). Ewaluacja od początku
na celu. Przy `--resume` czyszczony jest `ep_info_buffer`.

Weryfikacja przed startem: zmiana profilu w locie daje siłę @2 mm 2.50 / 8.75 / 15.00 N
dla frac 0 / 0.5 / 1 (oczekiwane identyczne); krótki bieg `train_board` — profil startowy
dociera do env, curriculum loguje `curriculum/estop_*`.

Config `ppo_board_6dof_estop_curr.yaml`: z `ft_sc1_v23_long`, f_start/f_peak/f_after
0/5/2 N → 10/20/5 N przez **4M**, potem 2M na celu (razem 6M), h = 1500, lr 1e-4,
target_kl 0.03, bez limitu. Katalog `logs/board_6dof/estop_curr`. Pierwsze 4 iteracje:
ep_len 455–598 (epizody kończone sukcesem), ep_rew ~0.59, approx_kl 0.012–0.019, 280 fps.
Biegi `estop_hw` i `estop_hw_lr` przerwane (nieważne, patrz sprostowanie).

**Wynik etapu A — curriculum siły okazało się zbędne.** Ewaluacja od początku na celu
(20 N, bez limitu, h = 1500, n = 20): 16.01M **19/20**, 16.11M 17/20, 16.21M 13/20, 16.31M 16/20,
16.41M 15/20, 16.51M 15/20, 16.61M 15/20, 16.71M reward −2.72 ± 8.78. Model po 100k krokach
(praktycznie bazowy) wciska przycisk 20 N przy skoku 6 mm. Dawne 0/20 przy 20 N dotyczyło
progu 12 mm (1700 N/m × 12 mm) — przeszkodą była głębokość, nie siła. W `estop_hw` ścianą
był więc limit 0.8727, nie przycisk. Trend w dół prawdopodobnie od treningu na miększym
przycisku niż ewaluacja (n = 20, w granicach szumu). Etap A przerwany przy 16.71M, `best_model`
= 16.01M.

**Etap B — curriculum limitu na przycisku 20 N** (start 2026-09-14 ~13:55). Config
`ppo_board_6dof_estop_lim.yaml`: `--resume logs/board_6dof/estop_curr/best_model/best_model.zip`,
limit 3.0 → 0.8727 liniowo przez 4M, +2M na 0.8727 (razem 6M), E-stop 20 N / 6 mm stały,
h = 2000, lr 1e-4, target_kl 0.03, ewaluacja od początku na 0.8727 + 20 N. Katalog
`logs/board_6dof/estop_lim`. Start zweryfikowany w logu: limit 3.0 we wszystkich env.
Wcześniejsze curricula limitu (`hwlim_curr`, `hwlim_curr3`) nie pomogły, ale szły na progu
12 mm — tamten wynik nie musi się przenosić.

**Przebieg etapu B do 19.71M (stan po 9 h 44 min).** Rollouty treningowe przy bieżącym
limicie (TensorBoard; sukces szacowany z długości epizodu: kończy go tylko sukces, h = 2000,
~600 kroków na sukces — szacunek zgrubny):

| krok | limit | śr. długość | szac. sukcesy | ep_rew |
|---|---|---|---|---|
| 16.03M | 2.99 | 905 | ~78 % | 0.33 |
| 16.81M | 2.57 | 1104 | ~64 % | −0.03 |
| 17.50M | 2.21 | 1436 | ~40 % | −0.81 |
| 17.89M | 2.00 | 1497 | ~36 % | −1.15 |
| 18.21M | 1.83 | 1676 | ~23 % | −1.98 |
| 18.53M | 1.66 | 1754 | ~18 % | −1.26 |
| 18.80M | 1.52 | 1853 | ~10 % | −1.87 |
| 19.12M | 1.34 | 1983 | ~1 % | −2.82 |
| 19.39M | 1.21 | **2000** | **0 %** | −2.46 |
| 19.71M | 1.03 | 2000 | 0 % | −2.67 |

Ewaluacja na docelowym 0.8727 + 20 N: **0/20 we wszystkich 37 punktach** (16.11M–19.71M).
Plateau ~35 % przy 2.2–1.9 było przejściowe. Poniżej ~1.9 skuteczność spada monotonicznie
do zera przy ~1.2 i nie wraca. Kryterium zapisane przy 2.0 („przy ~1.5 długość 1800–2000 =
curriculum nie wystarcza”) spełnione: 1.52 → 1853, 1.21 → 2000. **Curriculum limitu obalone
po raz trzeci**, tym razem na przycisku 6 mm — przyczyną nie była głębokość przycisku.
Uczenie stabilne przez cały bieg (approx_kl 0.01–0.03, std ~0.98): to granica zadania
w tej obwiedni, nie problem optymalizacji.

Wyniki: _decyzja o przerwaniu po konsultacji._

### 6.35. Skala obrotu OSC pod limit prędkości stawów (2026-09-14/15)

**Dlaczego.** Trzy podejścia pod limit 0.8727 (fine-tune, scratch, trzy curricula) dały zero,
a jedyną osią dławioną przez limit jest obrót (translacja 1.00x, obrót ~0.5x, 6.31–6.32).
`output_max` obrotu OSC_POSE = 0.5 rad: akcja 1.0 zadaje obrót celu o 0.5 rad w jednym
kroku 0.05 s. Limit obcina każdy staw osobno, co zmienia **kierunek** ruchu, nie tylko tempo.
Do tego szum eksploracji PPO (std ~1.0 w jednostkach akcji) sam w sobie zadaje obroty rzędu
0.5 rad/krok — polityka prawie nigdy nie widzi wykonalnego, małego obrotu. Mniejszy
`output_max` skaluje komendę i szum. Hipoteza, nie pewnik. Pomysł był zapisany 12–13.09
i odłożony po sprostowaniu „11x” (realna strata ~2x) — błąd kolejności.

**Implementacja.** `envs/controller_scale.py::with_osc_rot_output_max` (kopia configu
kontrolera z ±rot_max na częściach OSC_POSE), klucz YAML `osc_rot_output_max`, flaga
`compare_models --rot-max`. Dodatkowo `compare_models --estop-config <yaml>` bierze profil
E-stopu i próg z configu treningu.

**Pomiar** (`eval_2026-09-14/measure_rot_scale.py`, `rot_scale.csv`). Dla każdego kandydata
osobny env zbudowany ścieżką treningu, reset, `best_model` etapu A przez losowe 20–200
kroków, potem 6 kroków komendy testowej, **bez limitu** — ile prędkości stawów zada
sterownik. Szczyt |qvel| ramienia ze wszystkich substepów, 30 prób na tryb:

| rot_max [rad] | obrót: p50 / p95 [rad/s] | > 0.8727 | obrót/krok | obrót + transl. ±1: p95 | > 0.8727 |
|---|---|---|---|---|---|
| 0.50 | 1.67 / 2.29 | 100 % | 4.17° | 2.38 | 100 % |
| 0.20 | 0.90 / 1.65 | 57 % | 1.93° | 3.22 | 80 % |
| 0.12 | 0.79 / 1.47 | 40 % | 1.25° | 4.27 | 80 % |
| 0.08 | 0.76 / 1.35 | 30 % | 0.77° | 4.24 | 73 % |
| 0.05 | 0.68 / 1.45 | 33 % | 0.53° | 3.32 | 90 % |
| 0.03 | 0.65 / 1.18 | 20 % | 0.39° | 6.86 | 90 % |

Wersja 1 pomiaru (przywracanie stanu sim) była **nieważna** — wszystkie kandydaty
identyczne co do cyfry, bo OSC liczył cel z nieaktualnego stanu po `set_state`. Stąd
w wersji 2 kontrola, że obrót/krok maleje ze skalą (4.17° → 0.39°).

Wnioski:

- skala obrotu działa: 0.5 → 100 % prób ponad limitem, 0.2 → 57 %, 0.12 → 40 %;
- poniżej ~0.1 zysk zanika — zostaje dno ~20–33 % z innych przyczyn (prędkość
  odziedziczona po ruchu polityki przed testem, kontakty z tablicą);
- **translacja nasycona ±1 (0.05 m/krok) przekracza limit w 73–90 % prób niezależnie od
  skali obrotu** — to drugi kandydat, jeśli sama skala obrotu nie wystarczy (jedna zmienna
  naraz, dlatego teraz nie ruszana).

Wybór: **0.1 rad** — między zmierzonymi 0.12 (40 %) i 0.08 (30 %). Analitycznie OSC przy
kp 150 i tłumieniu krytycznym zadaje prędkość obrotu ≈ (kp/kd)·Δθ ≈ 6.1·0.1 ≈ 0.61 rad/s
wobec limitu 0.8727 — zapas na odwzorowanie na kilka stawów. Obrót 90° to ~4–5 s, przy
h = 2000 (100 s) bez problemu.

**Sim2real:** most musi dostać tę samą skalę (`policy_bridge.py --rot-max`). Stare modele
(0.5) i nowe (0.1) nie są wymienne.

**Test startu (reguła 2)** — `best_model` etapu A (trenowany przy 0.5) w skali 0.1, E-stop
20 N / 6 mm, h = 2000, n = 20 (`eval_2026-09-15/start_rot0.1_{nolim,hw}.csv`):

| warunki | sukcesy | wciśnięcia/ep | kolizje kroki/ep |
|---|---|---|---|
| skala 0.5, bez limitu (eval etapu A, h = 1500) | 19/20 | — | — |
| skala 0.1, bez limitu | 1/20 | 1.10 | 80.5 |
| skala 0.1, limit 0.8727 | 0/20 | 0.25 | 152.6 |

Stara polityka w nowej skali jest rozstrojona (obraca się 5x słabiej), ale sygnał nie jest
zerowy. Dlatego dwa etapy, jedna zmienna naraz, połączone skryptem
`logs/board_6dof/rot01_chain.sh` (start 2026-09-15 00:21):

- **C1** `ppo_board_6dof_rot01_nolim.yaml`: z `best_model` A, skala 0.1, **bez limitu**
  w treningu, 3M, h = 2000. Ewaluacja od początku w pełnej obwiedni (0.1 + 0.8727 + 20 N).
  Brak limitu w treningu zrealizowany przez `joint_vel_limits_start` 100 rad/s z anneal 1e12
  kroków (szczyty stawów < 10 rad/s, więc obcięcie nie działa), żeby ewaluacja mogła mieć
  limit docelowy — ten sam mechanizm co w 6.32, z odczytem zwrotnym.
- **C2** `ppo_board_6dof_rot01_hw.yaml`: z `best_model` C1, limit 0.8727 w treningu, 4M.
  Startuje tylko, jeśli C1 zapisał `ppo_board_final.zip` i `best_model`.

Pierwsze iteracje C1: ep_len ~1900 (prawie bez sukcesów, zgodnie z testem startu),
ep_rew −2.75 → −1.82, approx_kl ~0.02, ~285 fps.

**Wynik C1** (00:21–08:04, 16.01M → 19.01M). Rollouty treningowe bez limitu (sukces szacowany
z długości epizodu, h = 2000): ~10 % na starcie → ~18 % przy 17.2M → ~35 % przy 18.1M →
**~40 % przy 19.0M** — polityka dostraja się do skali 0.1, ale daleko od 19/20 etapu A przy 0.5.
Ewaluacja w obwiedni sprzętowej (0.1 + 0.8727 + 20 N, n = 20): **0/20 w 27 z 30 punktów**,
1/20 przy 17.91M, 18.01M i 18.31M. Skala obrotu 0.1 **nie** uczyniła limitu nieszkodliwym dla
polityki w tym stadium.

**Błąd w łańcuchu.** C2 startuje z `best_model` C1, a `EvalCallback` wybiera go po średnim
rewardzie ewaluacji. Przy 0 sukcesów to w praktyce szum: najlepszy reward (−0.54) padł przy
**17.11M**, gdy rollouty bez limitu miały ~18 %, a nie z końca C1 (~40 %). C2 wystartował więc
ze słabszego modelu (TensorBoard C2 zaczyna od kroku 17.127M). Reguła: przy zerowej
skuteczności ewaluacji `best_model` nie jest „najlepszym modelem” — start następnego etapu
brać z modelu końcowego albo wybierać po sukcesach.

**C2 po 1 h 37 min** (17.11M → 17.78M): rollouty z limitem ~0–1 % sukcesu (ep_len 1991–2000),
ewaluacja 0/20 we wszystkich 6 punktach. **Zatrzymany 2026-09-15 ~09:50 przy 17.78M**
— przejście do skali translacji (6.36).

### 6.36. Skala translacji OSC pod limit prędkości stawów (2026-09-15)

**Dlaczego.** Po C1 skala obrotu 0.1 nie wystarczyła (ewaluacja w obwiedni sprzętowej 0/20
w 27/30 punktów), a pomiar z 6.35 pokazał, że nasycona translacja ±1 (0.05 m/krok)
przekracza limit w 73–90 % prób niezależnie od skali obrotu. Jedna zmienna naraz: obrót
zostaje 0.1, zmieniana tylko translacja.

**Implementacja.** `with_osc_output_max(ctrl_cfg, pos_max, rot_max)` w
`envs/controller_scale.py`, klucz YAML `osc_pos_output_max`, `compare_models --pos-max`,
most `policy_bridge.py --pos-max` (POS_SCALE = 0.065 · pos_max / 0.05 — kalibracja Webotsa
z 6.29 była przy 0.05; Servo liniowo proporcjonalnie, niekalibrowane).

**Pomiar** (`eval_2026-09-15/measure_pos_scale.py`, `pos_scale.csv`) — metoda jak 6.35 v2,
skala obrotu 0.1, stany z modelu końcowego C1, 30 prób na tryb, bez limitu. Nowy tryb
**zero** (akcja 0) mierzy dno: prędkość odziedziczoną po ruchu polityki i od kontaktów.

| pos_max [m] | zero: > limit | transl ±1 (1 oś): > limit | przes./krok | transl ±1 (3 osie) + obrót ±1: p95 | > limit |
|---|---|---|---|---|---|
| 0.050 | 17 % | 37 % | 8.52 mm | 3.76 | **70 %** |
| 0.030 | 27 % | 30 % | 4.64 mm | 3.23 | **70 %** |
| 0.020 | 27 % | 30 % | 3.22 mm | 4.37 | **37 %** |
| 0.015 | 33 % | 37 % | 2.00 mm | 3.06 | 30 % |
| 0.010 | 23 % | 27 % | 1.84 mm | 3.73 | 33 % |
| 0.007 | 23 % | 27 % | 1.11 mm | 2.21 | 37 % |

Kontrola ważności: przesunięcie/krok maleje ze skalą (8.52 → 1.11 mm).

Wnioski:

- nasycona komenda spada z 70 % do poziomu dna przy **0.02**; mniejsze skale nic nie dodają;
- **dno ~17–33 % prób ponad limitem przy zerowej komendzie** — stawy nadgarstka (armature 0,
  damping 0.01) rozpędzają się od kontaktów i od ruchu sprzed testu. Tego nie usunie żadna skala;
  polityka musi nauczyć się ruchu, który tego unika, albo limit będzie ją nadal obcinał
  w ~1/4 sytuacji. Ogony p95 2–4 rad/s także przy zerze wskazują na kontakty.

Wybór: **0.02 m** — najmniejsza zmierzona skala, przy której nasycona komenda schodzi do dna,
z zachowaniem największej prędkości (3.2 mm/krok na jednej osi, ~64 mm/s).

**Test startu** — model końcowy C1 (trenowany przy translacji 0.05, obrocie 0.1) w skali
translacji 0.02, obrót 0.1, E-stop 20 N / 6 mm, h = 2000, n = 20
(`eval_2026-09-15/start_pos002_{nolim,hw}.csv`):

| warunki | sukcesy | wciśnięcia/ep | kolizje kroki/ep | stuck |
|---|---|---|---|---|
| bez limitu | 0/20 | 0.55 | 5.2 | 13 |
| limit 0.8727 | 0/20 | 0.15 | 99.0 | 12 |

Reguła decyzji zapisana przed testem: jeśli z limitem ≥ połowa wciśnięć bez limitu → trening
bezpośrednio w obwiedni sprzętowej; inaczej dwa etapy. 0.15 / 0.55 = 27 % → **dwa etapy**.
Limit nadal mocno szkodzi starej polityce (kolizje 5 → 99), choć ta jest rozstrojona o 2.5x
w translacji, więc to nie rozstrzyga hipotezy.

**Łańcuch** `logs/board_6dof/pos002_chain.sh` (start 2026-09-15 10:13):

- **D1** `ppo_board_6dof_pos002_nolim.yaml`: z **modelu końcowego** C1, translacja 0.02,
  obrót 0.1, bez limitu w treningu (jak C1), ewaluacja od początku na 0.8727, 3M, h = 2000;
- **D2** `ppo_board_6dof_pos002_hw.yaml`: z **modelu końcowego** D1 (poprawka błędu z 6.35),
  limit 0.8727, 4M. Startuje tylko, jeśli D1 zapisał `ppo_board_final.zip`.

**D1 po 4 h (19.03M → 20.61M, 1.6M kroków):** rollouty treningowe **bez limitu** — ep_len 2000
przez cały bieg (~0 % sukcesów), ep_rew ~−1.0 bez trendu; ewaluacja w obwiedni sprzętowej 0/20
we wszystkich 15 punktach. Dla porównania C1 (translacja 0.05) doszedł do ~40 % bez limitu.

**Czas biegu.** SB3 `time/fps` spadł z 294 do ~110, ale rollout nie zwolnił: iteracja stała
~64 s (257 fps) przez cały bieg. 15 ewaluacji po ~620 s = 2.33 h z 4.02 h (58 %) — 20 epizodów
× 2000 kroków bez sukcesów na jednym env (~72 kroki/s). Poprawka na kolejne biegi: ewaluacja
na kilku procesach albo rzadziej.

**Diagnoza: sufit siły OSC?** Siła docisku OSC ∝ kp · Λ · (odległość celu), a cel jest najwyżej
`output_max` od chwytaka — translacja 0.02 zamiast 0.05 obcina sufit siły ~2.5x (szacunek:
~40 N → ~16 N, poniżej szczytu 20 N). Test: model końcowy C1, translacja 0.02, obrót 0.1,
bez limitu, h = 2000, n = 20, te same ziarna (`start_pos002_soft5N.csv`):

| przycisk | sukcesy | wciśnięcia/ep | kolizje kroki/ep | stuck |
|---|---|---|---|---|
| E-stop 20 N | 0/20 | 0.55 | 5.2 | 13 |
| miękki 5 N (jak start etapu A) | 0/20 | 1.05 | 56.6 | 12 |

Siła odpowiada za część (wciśnięć 2x więcej na miękkim), ale **nie jest główną blokadą** —
nawet na miękkim przycisku bez limitu 0/20 i 12/20 zacięć. Polityka przy translacji 0.02
nie wykonuje zadania z powodu spowolnienia/rozstrojenia, a fine-tune przez 1.6M tego nie
odbudował. **Zmniejszanie skal akcji jako fine-tune: obrót 0.1 — polityka się dostraja (~40 %
bez limitu), ale bez zysku w obwiedni sprzętowej; translacja 0.02 — zadanie się rozpada.**

Wyniki: _decyzja o D1/D2 po konsultacji._

### 6.37. Limity prędkości według instrukcji Gen3 (User Guide R07) — sprostowanie 6.31 (2026-09-15)

Źródło: Kinova, *Gen3 Ultra lightweight robot User Guide*, R07
(https://www.kinovarobotics.com/uploads/User-Guide-Gen3-R07.pdf).
Kod drivera: `Kinovarobotics/ros2_kortex`, `kortex_driver/src/hardware_interface.cpp` (gałąź main).

**Limity stawów 6 DoF (s. 98–99):**

| tabela | kiedy obowiązuje | j1–j3 (duże) | j4–j6 (małe) |
|---|---|---|---|
| 46 — joint speed, limity ogólne | sterowanie wysokiego poziomu | **1.39 rad/s** (79.64 °/s) | **1.22 rad/s** (69.91 °/s) |
| 47 — joint speed | **tylko admitancja i sterowanie siłą** | 0.8727 rad/s (50 °/s) | 0.8727 rad/s |
| 48 — przyspieszenie, twarde | — | 5.2 rad/s² | 10.0 rad/s² |
| 49 — przyspieszenie | angular joystick, joint trajectory | 1.0 rad/s² | 10.0 rad/s² |

Tryby a stosowane limity (s. 99): Cartesian trajectory i **Cartesian joystick (twist) —
pozycja i prędkość stawów, bez przyspieszenia**; angular trajectory/joystick — także przyspieszenie.

**Zachowanie na limicie (s. 100):**

- tryby kątowe: prędkość stawu „max out at speed limit” (nasycenie per staw);
- **wszystkie tryby kartezjańskie: prędkości wszystkich stawów skalowane proporcjonalnie**, tak że
  żaden staw nie przekracza limitu **i zachowany jest zadany kierunek ruchu**;
- Cartesian joystick: normy prędkości narzędzia ograniczone do **0.5 m/s** i **0.8727 rad/s
  (50 °/s)** — komenda przeskalowana proporcjonalnie, kierunek zachowany.

**Tory sterowania w ros2_kortex:**

- kontrolery stawowe (`joint_trajectory_controller`) → `LOW_LEVEL_SERVOING`, driver wysyła pozycje
  wprost co cykl (`set_position`, bez ograniczania). Instrukcja (s. 92, 105): w trybie niskiego
  poziomu ochrony biblioteki sterowania **nie są stosowane**. Ograniczeniem jest fizyka aktuatora
  (maks. bez obciążenia 13 RPM = 1.36 rad/s duże, 25 RPM = 2.62 rad/s małe, s. 20) i monitory
  bezpieczeństwa (s. 123: duże ostrzeżenie 100 °/s, błąd 120 °/s; małe 180/200 °/s → **fault**,
  robot staje);
- `twist_controller` → `SINGLE_LEVEL_SERVOING` + `SendTwistCommand` = Cartesian joystick
  wysokiego poziomu: limity z tabeli 46 ze **skalowaniem proporcjonalnym** i kapy narzędzia
  0.5 m/s / 0.8727 rad/s;
- trajektorie wysokiego poziomu naruszające limity są **odrzucane**, nie przycinane (s. 102–103).

**Konsekwencje dla dotychczasowej pracy:**

1. **0.8727 rad/s dla stawów to limit admitancji/sterowania siłą, nie ogólny.** Wartości z PR #241
   (1.3963 / 1.2218) były zgodne z tabelą 46. Sprostowanie z 6.31 („PR #241 1.6x za luźny, właściwe
   0.8727 z joint_limits.yaml”) było **błędne** — `joint_limits.yaml` w ros2_kortex podaje wartość
   konserwatywną (tę samą co admitancja), używaną przez planer MoveIt.
2. **Forma limitu w symulacji była niewłaściwa dla trybu kartezjańskiego.** Hak `_limited_integration`
   przycina każdy staw osobno, co zmienia kierunek ruchu. Kinova w trybie twist skaluje wszystkie
   stawy proporcjonalnie i zachowuje kierunek.
3. Wszystkie wyniki „pod limitem 0.8727” (6.31–6.36: hwlim, hwlim_s1, hwlim_curr, hwlim_curr3,
   estop_hw, etap B, C1/C2, D1) liczone były w obwiedni **surowszej niż sprzęt** — co do wartości
   (1.6x ramię, 1.4x nadgarstek) i co do formy. Skale akcji z 6.35–6.36 (obrót 0.1, translacja 0.02)
   były obejściem tego modelu.
4. Otwarte: który tor zostanie użyty na robocie. Most (IK + JointTrajectory, także MoveIt Servo →
   JTC) idzie przez `LOW_LEVEL_SERVOING` — bez limitów Kinovy, z ryzykiem faultu przy szybkich
   ruchach. Tor twist (`twist_controller`) daje dokładnie opisane skalowanie i jest bezpieczniejszy.

**Implementacja limitu wg instrukcji** (2026-09-15; D1 zatrzymany przy 20.73M).
`ButtonBoardEnv`: `joint_vel_limit_mode` = `"clip"` (dotychczasowy, domyślny — stare wyniki
odtwarzalne) albo `"scale"` (proporcjonalnie, kierunek zachowany) oraz `eef_twist_limits`
= (lin m/s, ang rad/s). W haku `_limited_integration` po każdym podkroku: twist narzędzia
t = J·v (`mj_jacSite` na miejscu TCP), części liniowa i kątowa przeskalowane osobno do kapów,
powrót do stawów tłumionymi najmniejszymi kwadratami, potem skalowanie stawów do limitu,
q = q_prev + v·dt. Kinova skaluje komendę, a symulacja zrealizowaną prędkość — przybliżenie.
`compare_models --kinova-limits` = [1.39, 1.39, 1.39, 1.22, 1.22, 1.22] + tryb `scale` +
(0.5 m/s, 0.8727 rad/s). YAML: `joint_vel_limit_mode`, `eef_twist_limits`.
Weryfikacja: `eval_2026-09-15/verify_kinova_limits.py`.

**Weryfikacja fizyczna** (model A, E-stop 20 N, 3 epizody × 600 kroków z twardymi resetami,
ostatni podkrok każdego kroku sterowania, 1800 próbek):

| warunek | wynik | wymaganie |
|---|---|---|
| max \|v_i\| / lim_i po limicie | 1.000000 | ≤ 1 |
| min cos(v, v przed skalowaniem stawów) | 1.000000000000 | = 1 (kierunek zachowany) |
| max \|v narzędzia\| / max \|ω\| | 0.5000 m/s / 0.8727 rad/s | ≤ 0.5 / ≤ 0.8727 |
| max \|q − (q_prev + v·dt)\| | 0 | = 0 |

Skalowanie stawów aktywne w 17.3 % kroków, kapy narzędzia w **70.2 %**. Surowa prędkość
przekraczała limit stawu w 17.8 % kroków, a przycięcie per staw zmieniłoby kierunek ruchu
(cos < 0.999) tylko w 7.5 % — zniekształcenie kierunku nie było głównym mechanizmem szkody;
dominuje ograniczenie tempa, głównie przez kapy narzędzia. Regresja trybu domyślnego
(`clip`, k = 1700, limit 0.8727): obserwacje i nagrody identyczne bit w bit.

**Model A w obwiedni zgodnej z instrukcją** (E-stop 20 N / 6 mm, h = 2000, n = 20, te same ziarna,
`eval_2026-09-15/A_{nolim,kinova}.csv`):

| warunki | sukcesy | wciśnięcia/ep | czas sukcesu | kolizje kroki/ep |
|---|---|---|---|---|
| bez limitu | 15/20 | 3.80 | 21.2 s | 5.3 |
| limity wg instrukcji (tryb twist) | 1/20 | 1.15 | 24.6 s | 20.6 |
| dla porównania: 0.8727 per staw (`estop_hw`, eval 16.01M) | 0/20 | — | — | — |

Obwiednia zgodna ze sprzętem nadal odbiera modelowi większość skuteczności, ale w przeciwieństwie
do modelu 0.8727 per staw sygnał jest wyraźny (1.15 wciśnięcia/ep, 0 zacięć), co daje punkt
startowy do dotrenowania w tej obwiedni.

### 6.38. Dotrenowanie w obwiedni zgodnej z instrukcją Gen3 (2026-09-15)

**Uzasadnienie.** Model A (E-stop 20 N / 6 mm, bez limitu 15/20 przy h = 2000) w obwiedni
zgodnej z instrukcją (6.37) zachowuje wyraźny sygnał: 1/20 sukcesów, 1.15 wciśnięcia/ep, 0 zacięć.
Poprzednie punkty startowe w obwiedni 0.8727 per staw dawały 0–0.25 wciśnięcia/ep.

**Konfiguracja** `training/configs/ppo_board_6dof_kinova_ft.yaml`:

- start: `logs/board_6dof/estop_curr/best_model/best_model.zip` (16.01M);
- limity od kroku 0, bez kurikulum: stawy [1.39, 1.39, 1.39, 1.22, 1.22, 1.22] rad/s w trybie
  `scale`, kapy narzędzia 0.5 m/s i 0.8727 rad/s; domyślne skale OSC (0.05 m, 0.5 rad);
- E-stop 20 N / 6 mm, h = 2000, lr 1e-4, target_kl 0.03, `log_std_max` 0.0, 8M kroków;
- ewaluacja w tej samej obwiedni, 20 epizodów na **2 procesach co 200k** kroków (klucz
  `n_eval_envs` w `train_board.py`) — skrócenie narzutu ewaluacji opisanego w 6.36;
- bez kryterium przerwania: przełom z `LogStdClamp` (6.25) wymagał wielu milionów kroków,
  więc ocena trendu na podstawie ~1.5M kroków byłaby przedwczesna.

Start zweryfikowany w logu (`[limity] ... tryb=scale, eef_twist=[0.5, 0.8727]; ewaluacja: 2 env co
200000`). Pierwsze iteracje: ep_len ~1980–2000, ep_rew −1.86 → −1.68, approx_kl ~0.02, ~255 fps.

**Przebieg.** Bieg prowadzony w trzech częściach (przerwy na wyłączenie maszyny), wznawiany
z checkpointów 17 210 624 i 19 210 624; zatrzymany na **23 410 624** (z planowanych 24.01M),
bo trend był już rozstrzygnięty. Logi części: `train_part{1,2,3}.log`, kopie historii ewaluacji
`eval/evaluations_part{1,2,3}.npz` (`EvalCallback` nadpisuje plik przy każdym wznowieniu).

**Ewaluacja w obwiedni sprzętowej** (limity wg 6.37, E-stop 20 N / 6 mm, h = 2000, 20 epizodów
co 200k kroków, 35 punktów pomiarowych):

| faza | punkty | sukcesy łącznie | najlepszy punkt | ep_rew średnio |
|---|---|---|---|---|
| 16.21–17.21M | 6 | 1/120 | 1/20 | −1.23 |
| **17.41–19.01M** | 9 | **27/180** | **8/20 (17.41M)** | −1.28 |
| 19.41–23.21M | 20 | 13/400 | 2/20 | −1.28 |

Kolejne najlepsze punkty: 17.81M 5/20, 18.01M 4/20, 18.41M 3/20.

**Wynik.** Dotrenowanie w obwiedni zgodnej z instrukcją **przywraca sukcesy tam, gdzie wszystkie
warianty z limitem 0.8727 per staw dawały zero** — z 1/20 na starcie (model A, 6.37) do 8/20
w szczycie. Szczyt wypada ok. 1.4–2.0M kroków po starcie, po czym skuteczność opada do
~0.7/20 średnio i już nie wraca. Jest to ten sam wzorzec co w 6.25 (sufit ok. 16M, dalszy
trening degraduje politykę), tyle że w nowej obwiedni.

Uwaga do wyboru modelu: `EvalCallback` zapisuje `best_model` po **średnim rewardzie**, a nie po
liczbie sukcesów — przy tak zaszumionej ewaluacji wskazał 19.81M (rew −0.44), a nie 17.41M
(8/20). Model na sprzęt wybierany jest osobnym pomiarem na parach (n = 40, te same ziarna) —
kandydaci: 17.41M, 17.81M, model końcowy 23.41M oraz model A jako odniesienie.

**Pomiar rozstrzygający** (obwiednia sprzętowa: limity wg 6.37 + E-stop 20 N / 6 mm, h = 2000,
n = 40, te same ziarna dla wszystkich modeli; `logs/board_6dof/eval_2026-09-16/cand_*.csv`,
analiza `paired_candidates.py`):

| model | sukcesy | wciśnięcia/ep | czas sukcesu | kolizje kroki/ep | stuck |
|---|---|---|---|---|---|
| **checkpoint 17.41M** | **10/40 (25 %)** | 2.20 | 61.5 s | **65.8** | 0 |
| **checkpoint 17.81M** | **7/40 (18 %)** | 1.70 | 55.3 s | **10.8** | 1 |
| model końcowy 23.41M | 0/40 | 0.42 | — | 17.6 | 1 |
| A (punkt startowy) | 1/40 | 1.15 | 24.6 s | 24.8 | 2 |

Testy znaków na parach (sukcesy | wciśnięcia):

| para | sukcesy | p | wciśnięcia | p |
|---|---|---|---|---|
| 17.41M vs 23.41M | 10:0 | **0.00195** | 26:4 | **5.9e-05** |
| 17.41M vs A | 10:1 | **0.0117** | 19:10 | 0.136 |
| 17.81M vs 23.41M | 7:0 | **0.0156** | 19:4 | **0.0026** |
| 17.81M vs A | 7:1 | 0.0703 | 17:10 | 0.248 |
| 17.41M vs 17.81M | 8:5 | 0.581 | 16:14 | 0.856 |

**Wnioski.**

1. **Dotrenowanie w obwiedni zgodnej z instrukcją daje pierwsze polityki działające pod
   limitami sprzętowymi**: 25 % i 18 % skuteczności wobec 2 % modelu startowego
   (17.41M vs A: p = 0.0117). Wszystkie warianty z limitem 0.8727 per staw dawały 0.
2. **Degradacja po szczycie jest istotna statystycznie**: model końcowy przegrywa z obydwoma
   szczytami (10:0 i 7:0, p ≤ 0.016) i nie różni się od modelu startowego. Potwierdza to regułę
   z 6.25 — po osiągnięciu sufitu dalszy trening szkodzi.
3. **Oba szczyty są nieodróżnialne pod względem skuteczności** (8:5, p = 0.58), ale różnią się
   **kolizyjnością 6x**: 65.8 wobec 10.8 kroku/epizod. Kryterium bezpieczeństwa z 6.31
   (poniżej 20 kroków kolizji/ep) spełnia tylko **17.81M**, dlatego to on jest kandydatem na
   sprzęt, mimo niższej skuteczności.
4. Skuteczność w obwiedni sprzętowej (18–25 %) pozostaje daleko poniżej 76 % osiąganych bez
   limitów, a czas sukcesu rośnie z ~21 s do ~55–61 s. Transfer jest możliwy, ale kosztowny.

### 6.39. Próba przebicia sufitu w obwiedni sprzętowej — mniejszy `learning_rate` (2026-09-24)

**Uzasadnienie.** W 6.38 szczyt skuteczności (8/20 w ewaluacji, 10/40 w pomiarze na parach)
pojawił się ok. 1.5M kroków po starcie, po czym polityka degradowała się aż do 0/40 przy 23.41M.
Wzorzec zgodny z 6.25: po osiągnięciu sufitu dalszy trening szkodzi. Hipoteza sprawdzana tutaj:
wolniejsza aktualizacja pozwoli utrzymać i przesunąć szczyt zamiast go stracić.

**Konfiguracja** `training/configs/ppo_board_6dof_kinova_ft_lr.yaml`:

- start: `logs/board_6dof/kinova_ft/checkpoints/ppo_board_17810624_steps.zip` — kandydat na
  sprzęt z 6.38 (7/40, kolizje 10.8 kroku/ep);
- **jedyna zmienna: `learning_rate` 1e-4 → 3e-5**;
- bez zmian: limity wg instrukcji (stawy 1.39/1.22 rad/s w trybie `scale`, narzędzie 0.5 m/s
  i 0.8727 rad/s), E-stop 20 N / 6 mm, h = 2000, `target_kl` 0.03, `log_std_max` 0.0,
  ewaluacja 20 epizodów na 2 procesach co 200k kroków;
- 6M kroków (do ~23.8M), katalog `logs/board_6dof/kinova_ft_lr`.

Start zweryfikowany w logu: limity i tryb `scale` dotarły do środowisk, `learning_rate` 3e-5
wczytany z configu, `std` 0.975–0.978, approx_kl ~0.016, ~205 fps.

**Wynik: hipoteza potwierdzona — sufit z 6.38 przebity.** Bieg przeszedł pełne 6M kroków
(17.82M → 23.82M, ~12 h) bez degradacji. Ewaluacja w obwiedni sprzętowej (20 epizodów co 200k,
30 punktów):

| faza | punkty | sukcesy łącznie | najlepszy punkt |
|---|---|---|---|
| 18.0–19.8M | 10 | 51/200 (26 %) | 8/20 |
| 20.0–21.8M | 10 | 72/200 (36 %) | 12/20 |
| **22.0–23.8M** | 10 | **128/200 (64 %)** | **17/20 (22.01M i 23.61M)** |
| cały bieg | 30 | 251/600 (42 %) | — |

Porównanie z 6.38 (ten sam start, `learning_rate` 1e-4): tam szczyt 8/20 pojawił się raz,
~1.5M kroków po starcie, po czym skuteczność spadła do 0–2/20. Tutaj rośnie przez cały bieg.

Sygnały w treningu potwierdzają różnicę: szacowany odsetek sukcesów w rolloutach 0 % → ~46–49 %,
średnia długość epizodu 1998 → ~1310–1360 kroków, ep_rew −1.03 → ~−0.1, approx_kl stabilne
0.016–0.019. Kluczowa różnica jakościowa: **`std` malało 0.96 → 0.84** (polityka coraz
pewniejsza), podczas gdy w 6.38 rosło do 0.99 przy clampie 1.0, co towarzyszyło degradacji.

**Trend nie wygasł na koniec budżetu** — dwa ostatnie tysiące epizodów dają 13–17/20, więc sufit
w tej obwiedni nie został jeszcze osiągnięty. Wybór modelu i kolizyjność rozstrzyga pomiar na
parach (n = 40) w `logs/board_6dof/eval_2026-09-24/`: model końcowy 23.82M, checkpointy 22.01M
i 23.61M oraz poprzedni kandydat na sprzęt 17.81M jako odniesienie.

**Pomiar rozstrzygający** (obwiednia sprzętowa: limity wg 6.37 + E-stop 20 N / 6 mm, h = 2000,
n = 40, te same ziarna; `logs/board_6dof/eval_2026-09-24/`):

| model | sukcesy | wciśnięcia/ep | czas sukcesu | kolizje kroki/ep | stuck |
|---|---|---|---|---|---|
| **model końcowy 23.82M** | **32/40 (80 %)** | 4.03 | 42.5 s | **10.6** | 0 |
| checkpoint 22.01M | 29/40 (72 %) | 3.90 | 44.3 s | 12.3 | 0 |
| checkpoint 23.61M | 27/40 (68 %) | 3.50 | 32.5 s | 50.2 | 1 |
| poprzedni kandydat 17.81M (6.38) | 7/40 (18 %) | 1.70 | 55.3 s | 10.8 | 1 |

Testy znaków na parach: model końcowy vs 17.81M **1:26, p = 4.2e-07** (wciśnięcia 3:26,
p = 1.5e-05); 22.01M vs 17.81M 0:22, p = 4.8e-07; 23.61M vs 17.81M 1:21, p = 1.1e-05.
Trzy modele z tego biegu są między sobą nieodróżnialne (p = 0.27–0.79).

**Wniosek — zadanie rozwiązane w obwiedni sprzętowej.** Model końcowy osiąga **80 % przy
10.6 kroku kolizji na epizod**, czyli spełnia jednocześnie kryterium skuteczności i
bezpieczeństwa z 6.31 (> 40 %, < 20 kroków kolizji), których żaden wcześniejszy wariant nie
spełniał. Wciska średnio 4.03 z 5 przycisków, bez zacięć, w 42.5 s. Dla porównania model bez
żadnych limitów miał 76 % (h = 1500, miękki przycisk) — czyli **koszt obwiedni sprzętowej
zniknął**, a jedyną zmianą, która to umożliwiła, była poprawka modelu limitu (6.37) i
wolniejsze tempo uczenia (6.39).

**MODEL NA SPRZĘT: `logs/board_6dof/kinova_ft_lr/ppo_board_final.zip`** (23.82M).
Most uruchamiać z domyślnymi skalami OSC (`--pos-max 0.05 --rot-max 0.5`) i torem twist
(`twist_controller`), bo tylko tam firmware skaluje prędkości stawów proporcjonalnie (6.37).

Ponieważ trend na koniec biegu nadal rósł, przedłużenie treningu pozostaje otwarte.

### 6.40. Przedłużenie biegu w obwiedni sprzętowej, +6M (2026-09-25)

**Uzasadnienie.** Trend ewaluacji na koniec 6.39 nie wygasł (ostatnie punkty 13–17/20), a model
końcowy osiągnął 80 % w pomiarze na parach. Sprawdzany jest sam czas uczenia: czy dłuższy bieg
przy niezmienionych warunkach podnosi wynik dalej, czy trafia na sufit.

**Konfiguracja** `training/configs/ppo_board_6dof_kinova_ft_lr2.yaml`: start z
`logs/board_6dof/kinova_ft_lr/ppo_board_final.zip` (23.82M), **wszystkie parametry bez zmian**
wobec 6.39 (lr 3e-5, limity wg instrukcji, E-stop 20 N / 6 mm, h = 2000, `target_kl` 0.03,
`log_std_max` 0.0, ewaluacja 20 epizodów na 2 procesach co 200k), +6M kroków (do ~29.8M),
katalog `logs/board_6dof/kinova_ft_lr2`.

Start zweryfikowany w logu. Pierwsze iteracje potwierdzają poziom z końca 6.39: ep_len ~1120–1190
(wobec ~2000 na starcie 6.39), **ep_rew dodatni po raz pierwszy w tej obwiedni** (0.11–0.15),
`std` 0.808, approx_kl 0.020–0.022, ~230 fps.

**Wynik: dalszy wzrost, bez degradacji.** Pełne 6M kroków (23.82M → 29.82M, ~11 h). Ewaluacja
w obwiedni sprzętowej (20 epizodów co 200k, 30 punktów): **451/600 (75 %)** wobec 251/600 (42 %)
w 6.39. Rozkład w czasie:

| faza | sukcesy | najlepszy punkt |
|---|---|---|
| 24.0–25.8M | 121/200 (61 %) | 16/20 |
| 26.0–27.8M | 155/200 (78 %) | 18/20 |
| **28.0–29.8M** | **175/200 (88 %)** | **20/20 (28.02M i 28.62M)** |

Rollouty treningowe: ep_len 1174 → **609** kroków (szacunkowo ~99 % ukończonych epizodów),
ep_rew 0.19 → **0.686**, `std` 0.81 → **0.688** (dalszy spadek, bez śladu rozjeżdżania się),
approx_kl stabilne 0.021–0.026. Ostatnie cztery punkty ewaluacji (17, 15, 16, 15) mogą już
wskazywać plateau w okolicy 80–90 %.

**Pomiar rozstrzygający** (obwiednia sprzętowa, h = 2000, n = 40, te same ziarna;
`logs/board_6dof/eval_2026-09-25/`):

| model | sukcesy | wciśnięcia/ep | czas sukcesu | kolizje kroki/ep (mediana) | epizody bez kolizji | stuck |
|---|---|---|---|---|---|---|
| **model końcowy 29.82M** | 35/40 (88 %) | 4.55 | **23.4 s** | **1.0 (0.0)** | **21/40** | 3 |
| checkpoint 28.02M | 35/40 (88 %) | 4.53 | 24.8 s | 2.0 (0.5) | 20/40 | 1 |
| checkpoint 28.62M | **37/40 (92 %)** | 4.75 | 30.0 s | 3.8 (1.0) | 17/40 | 1 |
| poprzedni model na sprzęt 23.82M (6.39) | 32/40 (80 %) | 4.03 | 42.5 s | 10.6 (2.5) | 11/40 | 0 |

Testy znaków na parach wobec modelu z 6.39:

| miara | 28.02M | 28.62M | 29.82M |
|---|---|---|---|
| sukcesy | 3:6, p = 0.51 | 2:7, p = 0.18 | 4:7, p = 0.55 |
| kolizje | 8:23, **p = 0.011** | 12:21, p = 0.16 | 7:27, **p = 0.00082** |
| czas epizodu | 8:30, **p = 0.00047** | 8:31, **p = 0.00029** | 9:30, **p = 0.0011** |

**Wniosek.** Wzrost skuteczności 80 % → 88–92 % **nie jest istotny statystycznie** przy n = 40
(p = 0.18–0.55) — wymagałby ok. 100 epizodów. Istotnie poprawiły się natomiast dwie pozostałe
miary: **czas zadania skrócił się prawie dwukrotnie** (42.5 s → 23–30 s, p ≤ 0.001 dla wszystkich
trzech modeli) i **kolizyjność spadła ~10x** (10.6 → 1.0 kroku/epizod, p = 0.0008 dla modelu
końcowego; epizodów całkowicie bez kolizji 11/40 → 21/40). Trzy modele z tego biegu są między
sobą nieodróżnialne (p = 0.63–1.0).

**MODEL NA SPRZĘT: `logs/board_6dof/kinova_ft_lr2/ppo_board_final.zip`** (29.82M) — 35/40 (88 %)
przy najniższej kolizyjności (mediana 0 kroków) i najkrótszym czasie (23.4 s). Zastrzeżenie:
3 epizody zakończone zacięciem wobec 0 w modelu z 6.39; alternatywą o zbliżonych parametrach
i 1 zacięciu jest checkpoint 28.02M. Most: domyślne skale OSC (`--pos-max 0.05 --rot-max 0.5`),
tor twist (6.37).

**Potwierdzenie na 100 epizodach** (te same warunki i ziarna; `logs/board_6dof/eval_2026-09-25/n100/`):

| model | sukcesy (95 % CI Wilsona) | wciśnięcia/ep | czas sukcesu | kolizje kroki/ep (mediana) | epizody bez kolizji | stuck |
|---|---|---|---|---|---|---|
| checkpoint 28.62M | **94/100** [87.5–97.2 %] | 4.81 | 28.1 s | 3.6 (1.0) | 45/100 | 3 |
| **checkpoint 28.02M** | **92/100** [85.0–95.9 %] | 4.72 | **23.9 s** | **2.3 (1.0)** | **49/100** | 2 |
| model końcowy 29.82M | 87/100 [79.0–92.2 %] | 4.51 | 22.4 s | 5.0 (1.0) | 48/100 | 4 |
| poprzedni model 23.82M (6.39) | 74/100 [64.6–81.6 %] | 3.85 | 39.5 s | 23.7 (3.5) | 28/100 | 1 |

Testy znaków wobec modelu z 6.39 — wszystkie różnice istotne:

| miara | 28.02M | 28.62M | 29.82M |
|---|---|---|---|
| sukcesy | 4:22, **p = 5.3e-04** | 4:24, **p = 1.8e-04** | 6:19, **p = 0.015** |
| wciśnięcia | 4:25, **p = 1.0e-04** | 4:25, **p = 1.0e-04** | 6:20, **p = 9.4e-03** |
| kolizje | 20:62, **p = 3.7e-06** | 24:60, **p = 1.1e-04** | 21:65, **p = 2.2e-06** |
| czas epizodu | 15:81, **p = 4.0e-12** | 17:81, **p = 3.6e-11** | 17:76, **p = 4.5e-10** |

Modele z 6.40 nie różnią się między sobą istotnie (p = 0.12–1.0).

**Poprawka do wyniku n = 40.** Przy 40 epizodach najlepszy wyglądał model końcowy (kolizje
1.0 kroku/ep, mediana 0); przy 100 epizodach jego kolizyjność wynosi 5.0, a skuteczność 87 % —
najsłabszą z trzech. Kolejny przypadek reguły z 6.32: wynik na 20–40 ziarnach nie replikuje się
i nie może rozstrzygać wyboru modelu.

**MODEL NA SPRZĘT (ostatecznie): `logs/board_6dof/kinova_ft_lr2/checkpoints/ppo_board_28023552_steps.zip`**
— 92/100 [85–96 %], 4.72 wciśnięcia/ep, 23.9 s, 2.3 kroku kolizji/ep (mediana 1, połowa epizodów
bez kolizji), 2 zacięcia. Checkpoint 28.62M ma nominalnie wyższą skuteczność (94 %), ale różnica
jest nieistotna (2:4, p = 0.69), przy o 18 % dłuższym czasie i wyższej kolizyjności.

**Wynik dla pracy.** W obwiedni sprzętowej (limity prędkości wg instrukcji Gen3, E-stop 20 N
z progiem 6 mm, h = 2000) polityka osiąga **92 % skuteczności przy 2.3 kroku kolizji na epizod
i czasie 23.9 s**. Dla porównania punkt wyjścia tej ścieżki (model A, 6.37) miał w tych samych
warunkach 1/20, a wszystkie warianty liczone przy błędnym modelu limitu (6.31–6.36) — zero.

**Nagrania** (`docs/video/`, model 28.02M, te same warunki i ziarna co tabela n = 100; 1280×720,
20 fps = czas rzeczywisty przy `control_freq` 20 Hz):

- `model_na_sprzet_sukcesy.mp4` — cztery kolejne sukcesy, ziarna 1262 / 1253 / 1258 / 1244
  (211 / 213 / 226 / 231 kroków, 0–1 kroku kolizji), 44 s;
- `model_na_sprzet_porazka.mp4` — ziarno 1292, 3/5 przycisków, epizod zakończony zacięciem
  (jedno z 2 na 100), 100 s.

Liczby kroków zgadzają się co do jednego z CSV ewaluacji, co potwierdza parowanie ziaren między
`watch_board.py` a `compare_models.py`.

**`watch_board.py` dostał flagi obwiedni sprzętowej** (`--kinova-limits`, `--hw-limits`,
`--estop-config`, `--button-stiffness`, `--pos-max`, `--rot-max`, `--no-finger-gate`) oraz
`--seeds`, `--fps`, `--width`, `--height`. Stałe limitów są importowane z `compare_models.py`,
żeby nie istniały w dwóch miejscach (ta sama klasa błędu co 6.22/6.24). **Bez tych flag nagranie
pokazuje INNE zadanie niż pomiar** — miękki przycisk i brak limitów prędkości. Polecenie użyte do
nagrań jest w nagłówku `watch_board.py`.

Domyślny model w `ros2_bridge/policy_bridge.py` przestawiony z `ft_sc1_v23_long` (15.91M, uczony
bez limitów) na model na sprzęt 28.02M.

### 6.41. Test mostu ROS 2 w Webots dla modelu na sprzęt (2026-10-01)

Pierwsze uruchomienie modelu 28.02M na moście. Stos: Webots + `spot_launch.py` + `moveit_launch.py`,
Servo z `servo_launch.py`. Skrypty i logi: `logs/board_6dof/eval_2026-10-01/`.

#### Warstwa B1 (obserwacja + polityka) — POTWIERDZONA dla nowego modelu

| | wzorzec robosuite | most w Webots |
|---|---|---|
| `a[0:3]` | `[-1.0, -0.431, -1.0]` | `[-1.0, -0.43, -1.0]` |
| `\|a\|` średnie | 0.899 | 0.899 |
| nasycenie | 71 % | 71 % |
| dystans / align | 0.045 / 89.9° | 0.045 / 89.9° |

Wektor obserwacji zgodny **co do 0.17 mm w każdym bloku** (`b1_reference.py`, `b1_diff.py`,
`--dump-obs` w moście), 0 kroków z NaN. Warunek: wzorzec musi dostać te same stałe chwytaka co
most (`GRIPPER_*_MED`) — chwytak w robosuite drga (qvel do 78 rad/s), więc jego chwilowej wartości
nie da się porównywać krok do kroku. Bez podstawienia akcja zmienia się z `\|a\|` 0.899 na 0.804.

Sprostowanie oczekiwania: model uczony w limitach Gen3 **nie** komenderuje łagodniej. Nasycenie
to te same 71 %, a `\|a\|` wzrosło z 0.785 (ft_sc1_v23_long) na 0.899. Polityka żąda maksimum
i pozostawia skalowanie limiterowi.

#### Warstwa B2 (pętla zamknięta) — ZREALIZOWANY RUCH NIE ZALEŻY OD AMPLITUDY

Miara po obu stronach identyczna: `dp` = przyrost pozycji EEF, `dR` = kąt geodezyjny orientacji,
mediana na krok. Wzorzec robosuite z tego samego stanu startowego (`b2_robosuite_trace.py`):
**7.92 mm i 2.35°/krok**, align 89.9° → 20.7°, 3/5 przycisków w 300 krokach.

| bieg | droga | zmienna | `dp` | `dR` | align po 300 |
|---|---|---|---|---|---|
| C | Servo | horyzont 10 ms, 1.0 m/s | 0.15 | 0.03 | 83.4° |
| A | Servo | + 5 podkroków | 0.33 | 0.03 | 83.4° |
| B | Servo | + tempo 0.158 m/s / 0.82 rad/s | 0.20 | 0.04 | 79.7° |
| D | Servo | `publish_period` 0.05 (param set) | 0.25 | 0.06 | 78.9° |
| — | IK | `POS_SCALE` 0.065, horyzont 0.02 s | 36.9 na starcie, potem 0 | 8.14 | 33.1°, blokada stawu |
| E | IK | `POS_SCALE` 0.0139 (÷4.66) | 15–25 na starcie, potem 0 | — | 60.7°, blokada |
| **F** | IK | **horyzont 0.05 s = okres sterowania** | **1.48 śr.** | **0.45 śr.** | **23.6°** |

Obalone pomiarem, po kolei: amplituda komendy (zmniejszenie 4.66x **nie zmniejszyło** ruchu —
15–25 mm/krok przy zadanych 13.9), liczba podkroków (2.2x), dławienie kolizyjne
(`collision_velocity_scale` = 1.0 przez cały bieg), `publish_period` ustawiany na żywo
(zrzut komend pokazuje `time_from_start` 10 ms mimo raportowanych 0.05 — Servo czyta ten
parametr raz, przy starcie).

**Przyczyna wspólna: horyzont punktu trajektorii krótszy od okresu sterowania.** IK wysyłał
`time_from_start` 0.02 s, Servo 0.01 s, a komendy idą co 0.05 s. Kontroler dostaje cel
nieosiągalny w zadanym czasie, jedzie z maksymalną prędkością i zostaje **wywłaszczony** następną
komendą. Zrealizowany ruch przestaje zależeć od amplitudy — zależy od tego, ile kontroler zdąży.
To tłumaczy dwa pozornie sprzeczne objawy jednym mechanizmem: IK pędzi 4.7x za szybko i ucieka
ramieniem na 0.24 m do blokady stawu, Servo pełza 50x za wolno i staje na singularności.
Servo publikuje przy tym punkty **bez prędkości** (`velocities: []`), czyli kontroler nie dostaje
nawet informacji, jak szybko ma jechać.

**Bieg F potwierdza tezę jakościowo, nie ilościowo.** Horyzont 0.05 s plus skale dobrane tak, by
ZADANA delta równała się tej, którą robosuite REALIZUJE (9.3 mm, 2.76° przy `\|a\|`=1): średnie
`dp` rośnie 0.28 → **1.48 mm** (5.3x), `dR` 0.04 → **0.45°** (11x), kroki bez ruchu spadają
78 % → 33 %, a **align dochodzi do 23.6 %** wobec 20.7° w robosuite — manewr obrotu nadgarstka
wykonuje się po raz pierwszy. Ramię przestaje uciekać. Zostaje luka tempa 5.4x i jedna trzecia
kroków stojących (podejrzenie: nieudane `compute_ik`, nieinstrumentowane).

#### Test osi: znak i kierunek realizacji akcji (`--axis-test`)

Luka metodologiczna: **B1 porównuje sam wektor akcji, nie jego interpretację.** Zgodność co do
0.001 nie mówi nic o tym, czy `a[0] = -1` znaczy w Webots ten sam kierunek co w robosuite.
Test: stała akcja ±1 na każdej z 12 osi przez 20 kroków, pomiar przesunięcia punktu chwytu
i kąta geodezyjnego orientacji. Skrypty: `--axis-test` w moście, `axis_robosuite.py`.

**Wszystkie 12 osi zgodne co do znaku i osi dominującej.** Odwrócenia ramki nie ma.
Wyszły natomiast trzy rozbieżności, każda istotna:

| | most | robosuite |
|---|---|---|
| translacja na oś | 4.6–5.0 mm/krok, **izotropowa** | 1.7 / 6.7 / 12.9 mm/krok (X / Y / Z) |
| +X od pozy home | 97 mm swobodnie | **33.9 mm — opór tablicy** |
| przesunięcie chwytaka przy obrocie ±RX | **14.9 mm** przy 27.8° | **350 mm** przy 45.1° |

1. **W Webots nie ma przeszkody.** Przyciski nie mają fizyki (6.24), ale konsekwencja jest
   szersza: ramię przejeżdża przez tablicę na wylot. Wszystkie pomiary tempa z 6.41 robione są
   w scenie bez przeszkody, której polityka uczyła się unikać.
2. **Mechanizm Webotsa jest izotropowy, OSC nie.** Jedna stała `POS_SCALE` nie może trafić
   w trzy różne wzmocnienia osi — to ogranicza dokładność kalibracji z góry.
3. **Obrót realizuje się wokół innego punktu.** Po normalizacji na kąt most daje ~15 mm
   przesunięcia chwytaka tam, gdzie robosuite daje ~215 mm (**14x mniej**). W robosuite obrót
   nadgarstka zamiata chwytak po łuku o promieniu rzędu `GRIP_OFFSET`; na moście chwytak obraca
   się niemal w miejscu. To dotyczy **głównego kanału tej polityki** — zadanie jest rozwiązywane
   obrotem nadgarstka (align 89.9° → 20.7°), a nie dojazdem. Hipoteza: różny punkt obrotu
   (`GRIP_OFFSET` wzdłuż osi Z pozy docelowej) albo niezdolność OSC do utrzymania pozycji przy
   skalowaniu prędkości stawów. NIEROZSTRZYGNIĘTE.

Obrót na moście idzie też 0.57x wolniej (1.35–1.44 wobec 2.25–2.48 °/krok). Próba nadrobienia
tego mnożnikiem `ROT_SCALE` ×1.7 **pogorszyła wynik** (mediana `dp` 7.24 → 0.38, kroki stojące
19 % → 31 %), więc nie jest to brakująca stała.

#### Kalibracja skali przy poprawnym horyzoncie

Po ustawieniu horyzontu 0.05 s most realizuje ~19 % zadanej komendy (OSC w robosuite ~26 %).
Sweep `POS_SCALE`, po 2 powtórzenia: **0.0279 trafia w tempo robosuite** — mediana `dp`
7.24 i 7.40 mm wobec 7.92, powtórzenia zgodne co do 2 %. Przy 0.0186 jedno powtórzenie
zdegenerowało: ramię zaklinowało się w pozie, z której `compute_ik` zwraca **NO_IK_SOLUTION
w 170/300 kroków** — most przestaje wtedy wysyłać cokolwiek i układ nie ma jak wyjść ze stanu
absorbującego. W pozostałych biegach 0/300 błędów IK.

**Rozrzut między biegami.** Ta sama konfiguracja, trzy powtórzenia pod rząd: średnie `dp`
2.80 / 2.87 / 2.39 mm (±10 %). Ale pojedyncze biegi tej samej konfiguracji puszczane w innych
momentach dały 1.48 i 4.99. **Porównywać warianty wolno tylko w jednej serii.** Wcześniejszy
wniosek „horyzont daje 5.3x" jest zawyżony; uczciwie: przesuwa most z 0.28 na ~2.6 mm, czyli
rząd wielkości, i tego rozrzut nie unieważnia. Kroki, w których IK zwraca rozwiązanie, a ramię
nie drga: 19–43 % zależnie od biegu — to kontroler, nie kinematyka.

#### Zmiany w narzędziach

`policy_bridge.py`: `--dump-obs`, kolumny `dp`/`dR` z medianami w podsumowaniu, `--ik-horizon`,
`--exec-pos-scale`/`--exec-rot-scale` (kalibracja mechanizmu — świadomie ODDZIELONE od
`--pos-max`/`--rot-max`, które mówią, czym trenowano model), `--servo-substeps`,
`--servo-vel-lin`/`--servo-vel-rot` (knoby wprost w m/s i rad/s). Domyślny model przestawiony
na 28.02M. `TIME_FROM_START` 0.02 zostaje domyślne do czasu rozstrzygnięcia luki tempa.

Pułapka środowiskowa: `/joint_states` podaje stawy w kolejności `1,2,4,5,3,6` (most przestawia
po nazwach). Druga: `set -u` wywraca `source /opt/ros/humble/setup.bash`.

### 6.42. Geometria sceny i poza startowa: korpus Spota, kamera, odległość postoju (2026-10-02/04)

Kampania wywołana trzema obserwacjami użytkownika, z których **żadnej nie dało się zobaczyć
w dotychczasowych metrykach**. Wynik końcowy: **100/100** w scenie wierniejszej niż ta,
w której mierzono 92/100 z 6.40.

#### Co było nie tak ze sceną

**1. Korpus Spota i kamera nie istniały w treningu.** Zmierzone z `protos/Spot.proto`
(suma AABB 16 pudełek `BODY_BO` z uwzględnieniem obrotów sześciu z nich) i z `Kinect.proto`
(wariant `foot FALSE` → geometria kolizyjna to listwa sensora 0.28 × 0.038 × 0.064, nie klocek
0.084 ze stojaka). Montaż z TF: `base_link → gen3_base_link` = [0.256, 0, 0.112], rotacja
jednostkowa. W ramce podstawy ramienia korpus zajmuje X −0.729…+0.266, Y ±0.139,
Z −0.202…+0.002; kamera X +0.209…+0.273, Y ±0.140, Z −0.026…+0.012.

Koszt tej luki (model 28.02M, n=40, te same ziarna):

| scena | sukcesy | kolizje kroki/ep | kontakt ze Spotem |
|---|---|---|---|
| bez Spota (tak uczono model) | 35/40 (88 %) | 2.0 | — |
| + korpus | 21/40 (52 %) | 37.0 | 293 kroku/ep, 40/40 ep. |
| + korpus + kamera | 20/40 (50 %) | 89.5 | 446 kroku/ep, 40/40 ep. |

**2. Kamera nadgarstkowa nie widziała tablicy.** Oś narzędzia w pozie startowej to
[0.002, 0, −1] — chwytak pionowo w dół. Kąty do przycisków 48.8–120.6° przy połowie FOV 32.5°,
czyli **0/5 w kadrze, górne przyciski za kamerą**. Percepcja na sprzęcie (ArUco → nadzorca →
`target_relative_pos`) nie miała z czego wystartować; w symulacji niewidoczne, bo env podaje
cel z informacji uprzywilejowanej.

**3. Dolne przyciski.** Z bryłą Spota btn3 spadał z 38/40 na 21/40 — dolne gniazda (z = 0.538)
leżą 8–9 cm **poniżej górnej krawędzi platformy** (0.622), więc dojazd wymagał nurkowania
tuż nad listwą kamery.

#### Dlaczego kara nie wystarczyła, a geometria tak

Dwa etapy dokręcania `spot_collision_cost` (0.13 → 5.0, czyli zrównanie z karą za tablicę):

| model | sukcesy | kolizje | kontakt ze Spotem |
|---|---|---|---|
| 28.02M (przed) | 50 % | 89.5 | 446.5 — 39/40 ep. |
| etap 1, +2M (kara 0.13) | 65 % | 50.1 | 151.7 — 38/40 ep. |
| etap 1, +4M | 55 % | 15.0 | 260.1 — 39/40 ep. |
| etap 1b, 38.92M (kara 5.0) | 50 % | 42.1 | 75.5 — 15/40 ep. |

Kara obniżyła kontakt 5.9x, ale kosztem skuteczności, a gładkość pogarszała się przez całą
kampanię (1.247 → 1.68). **Odsunięcie Spota o 24 cm dało kontakt 0.0 i kolizje 0.2 od razu** —
to, czego kara nie potrafiła przez 9M kroków.

Pomiar dostępu wg odległości postoju (manipulowalność `sqrt(det(J Jᵀ))` w pozie wciskania,
najgorszy przycisk; zasięg Gen3 6DOF ~0.90 m):

| `base_x_offset` | dystans bazy | manipulowalność | zapas limitów | kąt btn0 |
|---|---|---|---|---|
| −0.10 (dotychczas) | 0.523 m | — | — | 64.1° (w dół) |
| −0.18 | 0.603 m | 0.0234 | 0.050 rad | — |
| −0.26 | 0.683 m | 0.0357 | 0.192 rad | 21.1° |
| **−0.34** | 0.763 m | **0.0502** | **0.389 rad** | **18.0°** |

**Dostęp ROŚNIE z odległością.** Z bliska ramię musi się składać i pracuje blisko osobliwości.
Wcześniejsze przypuszczenie, że 87 % zasięgu musi być złe, zostało obalone pomiarem.

#### Poza startowa: stabilność jako twarde kryterium

Pozy w pełni poziome (align ≈ 0, 5/5 w kadrze) **istnieją, ale OSC ich nie utrzymuje** — dryf
0.35–2.69 rad w 30 krokach zerowej akcji, wszystkie w gałęzi z `j1 ≈ −3.04` (podstawa obrócona
o 180°). Kryterium stabilności dodane do wyszukiwania pozy; bez niego bieg startowałby od
upadku ramienia.

**Asymetria z przechyłu podstawy.** Pierwsza wybrana poza miała `j1 = −0.4` — na granicy
narzuconej optymalizatorowi. Skutek po 2M kroków: btn0 13/40, btn1 15/40, btn2 14/40,
btn3 8/40, **btn4 0/40**, przy czym btn3 i btn4 leżą lustrzanie. Sukces wymaga kompletu, więc
jedno gniazdo zerowało całą skuteczność mimo czterech wciśnięć w 6/40 epizodów.
Po ustawieniu `j1 = 0` (asymetria kątów 0.01°) **btn4 wrócił do 39/40 po 2M kroków**.

Kryterium kadru zluzowane świadomie: jeden znacznik ArUco daje pozę 6D, a układ gniazd
omniboarda jest znany, więc jedna detekcja tablicy wystarcza do wyliczenia wszystkich pięciu
(tak działa `detect_dex_board.py`). Nie trzeba 5/5 w kadrze.

#### Wynik

`init_qpos: [0.0, 1.4692, -2.2410, 3.1403, -2.0200, 3.1416]`, `base_x_offset: -0.34`.
Zweryfikowane przy pełnym szumie: align 8.4° (było 89.9), dryf 0.0171 rad, kontakt przy
starcie 0/4, 3/5 przycisków w kadrze, dystans do celu 0.165 m.

**MODEL: `logs/board_6dof/spot_e2b/checkpoints/ppo_board_50223552_steps.zip`** — n=100,
pełna scena sprzętowa (korpus + kamera + limity wg instrukcji + E-stop 20 N/6 mm, h = 2000):

| | +8M (50.22M) | +4M (46.22M) |
|---|---|---|
| sukcesy | **100/100** [96.3–100 %] | 99/100 |
| czas sukcesu | **7.4 s** | 13.0 s |
| epizody z kolizją | 7/100 | 31/100 |
| kroki kolizji/ep | **0.2** | 0.9 |
| kontakt ze Spotem | 2/100 ep., 0.0 kroku | 0/100 ep., 0.0 kroku |
| wciśnięcia per przycisk | 100/100 każdy | 99–100/100 |
| zacięcia | 0 | 0 |

**ZASTRZEŻENIE: to INNA scena niż 92/100 z 6.40** (inny montaż, inna poza). Tych liczb nie
wolno zestawiać jako postępu polityki. Właściwe porównanie to **20/40 → 100/100** tej samej
polityki w scenie z korpusem Spota. Zysk pochodzi w dużej mierze z geometrii, nie z uczenia,
a zadanie stało się naraz łatwiejsze i bardziej realistyczne.

#### Zmiany w narzędziach i dwie usterki tej samej klasy

`ButtonBoardEnv`: `spot_body`, `spot_front_cam`, `spot_collision_cost`, `init_qpos`.
`compare_models`/`watch_board`: `--spot-body`, `--spot-front-cam`, `--base-x-offset`,
`--init-qpos`, `--action-ema`.

**Geomy bez atrybutu `group` trafiają do grupy 0 (kolizyjnej), której renderer robosuite NIE
RYSUJE.** Bryła Spota działała fizycznie, ale była niewidoczna na zrzutach — a tak samo
**fizyczna ściana za tablicą od 2026-08-05**. Obie naprawione (`group="1"`); czysto wizualne.

`compare_models` **w ogóle nie miał `base_x_offset`** — każda ocena szła z domyślnym −0.10.
Bez poprawki model uczony na −0.34 byłby mierzony w innej scenie. To ta sama klasa błędu co
6.22/6.24: parametr istnieje w środowisku, ale narzędzie go nie przekazuje.

#### Filtr akcji: bang-bang jest nośny (zamknięte)

`--action-ema` w `compare_models`, model 28.02M, n=40, obwiednia sprzętowa:

| α | sukcesy | czas | kolizje kroki/ep | `‖Δa‖` |
|---|---|---|---|---|
| bez filtru | 35/40 (88 %) | 24.8 s | 2.0 | 1.344 |
| 0.5 | 31/40 (78 %) | 40.9 s | 12.9 | 0.609 |
| 0.3 | 18/40 (45 %) | 52.4 s | **111.4** | 0.340 |
| 0.2 | 8/40 (20 %) | 64.7 s | 95.0 | 0.247 |

**Gładkości nie da się dołożyć po treningu.** Przy ‖Δa‖ 0.34 kolizyjność rośnie 56x — polityka
hamuje przed tablicą właśnie przeskokiem komendy. Etap gładkości (`action_rate_cost` w górę)
NIE został uruchomiony; gładkość modelu to nadal 1.317.

#### Otwarte

`j5` siedzi na granicy −2.02 (limit −2.10) narzuconej optymalizatorowi — ten sam rodzaj
ograniczenia, który przy `j1` dał asymetrię. Zwolnić przy następnej iteracji.
Trening zatrzymany na 52.23M z 54.22M (plateau nagrody 1.05 przez 10 punktów + ryzyko OOM
przy dwóch ocenach n=100 obok treningu 8-env).

### 6.44. Twarde limity przyspieszeń Gen3 — najtwardsze ograniczenie transferu (2026-10-05/06)

#### Źródło, zweryfikowane u źródła

Instrukcja Gen3 R07, s. 99–100, **sekcja Gen3 6 DoF** (tabele 45–50):

| tabela | limit | dotyczy |
|---|---|---|
| 48 — **twarde** | j1–3 **5.2 rad/s²** (297.94 °/s²), j4–6 10.0 | **zawsze** |
| 49 | j1–3 **1.0 rad/s²**, j4–6 10.0 | **tylko** angular joystick i joint trajectory |

Tabela trybów (s. 100): **Cartesian joystick, czyli nasz tor twist, ma nakładane wyłącznie limity
pozycji i prędkości — przyspieszeń NIE.** Zachowanie przy limicie: „joint acceleration will max out
at acceleration limit" — robot **przycina, nie faultuje**; skutkiem jest opóźnienie wobec zadanej
trajektorii, czyli problem wierności, nie bezpieczeństwa.

**⚠ Mierząc zgodność używać 5.2/10.0, NIE 1.0/10.0.** Pomiar przeciwko 1.0 zawyża naruszenia:
dla tego samego modelu 94.9 % wobec 72.3 %.

#### Dwie usterki pomiarowe znalezione po drodze

**Kolejność w haku limitów.** Clip przyspieszeń był **pierwszy**, a kapy twist i limit prędkości
zmieniały prędkość **po nim** — i potrafiły zmienić ją o więcej, niż przyspieszenie dopuszcza.
Zmierzone: j1 **33.9 rad/s²** przy limicie 5.2, czyli sześciokrotnie. Clip przeniesiony **na koniec**
(bezpiecznie także dla pozostałych limitów: `v_prev` i `v` są oba w obwiedni, a punkt pomiędzy nimi
też). Po poprawce **0.00 %** przekroczeń we wszystkich konfiguracjach, maksima dokładnie 5.2/10.0,
zweryfikowane także na 200 krokach **losowych** akcji.

**Licznik naruszeń bez tolerancji.** Porównanie `> 5.2` na wartościach **równych** 5.2 liczyło
nasycenia jako naruszenia (59.8 % przy idealnie działającym limicie). Dodana tolerancja 0.1 %.

#### Koszt limitu i co dał trening (n=40, pełna obwiednia, te same ziarna)

| | sukcesy | czas | kolizje kroki/ep (mediana) |
|---|---|---|---|
| **bez limitu** (58.12M, n=100) | 100/100 | 5.5 s | 0.4 |
| z limitem, **bez przetrenowania** | 38/40 (95 %) | 25.0 s | 67.2 |
| + 1.5M treningu | 39/40 | 31.9 s | 52.9 |
| + 3M | 33/40 | 27.8 s | 36.0 |
| + 6.8M | 37/40 | 26.4 s | 32.3 |
| **+ 8M (koniec etapu 4)** | **39/40 (98 %)** | **18.8 s** | **18.7 (12)** |

Pary, model przed i po treningu, oba z limitem: kolizje **30:10, p = 0.0022**, mediana 38 → 16.
Czas i skuteczność bez zmian. **Trening działa**, ale poziom docelowy (< 5) pozostaje odległy.

**⚠ SPROSTOWANIE do 6.43.** Wynik **100/100 w 5.5 s opierał się na przyspieszeniach do 15–27 rad/s²**
(p90–p99), czyli na ruchach, których prawdziwy Gen3 **nie wykona** — tabela 48 to granica możliwości
aktuatorów, nie nastawa. Uczciwa liczba na sprzęt to **39/40 przy 18.8 s i 18.7 kroku kolizji**.

#### Momenty — sprawdzone, czyste

Tabela 50 (39 N·m dla j1–3, 9 N·m dla j4–6): **0 % przekroczeń** w obu modelach, bo model robota
w robosuite ma te limity wpisane i aktuatory się na nich nasycają. Ale model uczony z limitem
przyspieszeń pracuje **znacznie bliżej ściany**: p99 na barku 26.3 i 31.1 N·m wobec 4.7 i 20.8 bez
limitu, maksima dobijają do 39.0. Na sprzęcie oznacza to brak zapasu na tarcie, luzy i różnice masy.

### 6.45. Kara za zbyt szybki dojazd — WYNIK NEGATYWNY (2026-10-06)

**Hipoteza.** Polityka hamowała przed tablicą gwałtownym przeskokiem komendy; limit przyspieszeń
ten hamulec obcina. Istniejące człony nie adresują tego: `collision_cost` i `contact_speed_cost`
odpalają **po** kontakcie, `action_rate_cost` karze szarpanie **wszędzie**. Dodany człon karze
składową prędkości **w stronę celu** ponad `v_dop = dystans / tau` — „miej tyle prędkości, żeby
zdążyć się zatrzymać". Cofanie i ruch boczny bez kary (żeby nie dławić obrotu nadgarstka).
Geometria **prawdziwa** (`site_xpos`), nie obserwacja z biasem percepcji.

**Kalibracja z pomiaru, nie na oko:** przy koszcie 1.0 kara wynosiłaby 0.005 nagrody (nic).
tau 0.20 → 5.3 % kroków z nadmiarem, 0.30 → 11.1 %, 0.45 → 19.1 %; mediana nadmiaru ~0.11 m/s.
Wybrane tau 0.30 i koszt 50 → **16 % nagrody epizodu**, zweryfikowane (+0.9615 → +0.8019).

**Wynik: brak efektu.** 8M kroków, n=40, pary wobec końca etapu 4:

| | kolizje | misuse | czas | sukcesy |
|---|---|---|---|---|
| e5 +4M | 20:20, p=1 | 7:6, p=1 | 20:20, p=1 | 3:1, p=0.63 |
| e5 +8M | 21:19, p=0.88 | 7:3, p=0.34 | 21:19, p=0.88 | 2:1, p=1 |

Żadna miara istotna. **Człon do wycofania** (`approach_speed_cost` zostaje w env, domyślnie 0.0).

#### Lekcja metodologiczna: średnia kontra mediana

Kolizyjność ma **ciężki ogon** — średnia jest zdominowana przez pojedyncze złe epizody. Średnie
w tej kampanii skakały 18.7 → 12.8 → 20.9, a mediany szły gładko 12 → 10 → 9. Porównywanie
checkpointów po średnich dało **dwa błędne wnioski**: „krzywa się wypłaszczyła" przy +6.8M etapu 4
(a ostatnie 1.2M zdjęło −13.6) oraz „kara nie działa" przy +2M etapu 3 (a bieg dowiózł poprawę).
**Reguła: dla miar o ciężkim ogonie porównywać medianę i test na parach, nigdy samą średnią.**

#### Polityka NIE zinternalizowała obwiedni — nauczyła się na niej polegać

Test: ten sam model **bez** ogranicznika. Jeśli trzyma się poniżej 5.2 sam z siebie — nauczył się
obwiedni; jeśli nadal przekracza — polegał na ograniczniku.

| model | bez ogranicznika | z ogranicznikiem | mediana j1–j3 (bez) |
|---|---|---|---|
| przed etapem 4 (58.12M) | **62.0 %** kroków ponad 5.2 | 0.0 % | 2.28 / 2.16 / 3.37 |
| **po etapie 5 (74.12M)** | **74.8 %** | 0.0 % | **3.46 / 3.20 / 4.26** |

**Po 16M kroków treningu WEWNĄTRZ limitu polityka żąda przyspieszeń ponad limit CZĘŚCIEJ niż
przedtem** (74.8 % wobec 62.0 %), a mediany wzrosły o połowę. Ogranicznik stał się częścią obiektu
sterowania: polityka komenderuje mocniej, bo i tak zostanie przycięta.

**Konsekwencja dla sprzętu.** Ramię nie przekroczy limitów, bo nie może — w symulacji przycina hak,
na sprzęcie firmware („acceleration will max out at acceleration limit"). Wynik 38/40 **pozostaje
ważny**, ale z zerowym zapasem: polityka pracuje na nasyceniu przez trzy czwarte kroków i jest
dostrojona do **naszej konkretnej semantyki przycinania** (clip prędkości ku `v_prev` w każdym
podkroku + korekta pozycji). Firmware Kinovy ogranicza po swojemu, prawdopodobnie na poziomie
momentów i z własnym filtrowaniem — każda różnica zmieni zrealizowany ruch, a polityka nie ma
marginesu, żeby to wchłonąć.

**Wniosek ogólny: trenowanie wewnątrz twardej obwiedni nauczyło politykę EKSPLOATOWAĆ nasycenie,
a nie go unikać.** Obie kary, które miały temu zapobiec, zawiodły lub zadziałały słabo:
`action_rate_cost` (6.43) dał −23 % ‖Δa‖, `approach_speed_cost` (6.45) nie dał nic.
Do sprawdzenia: randomizacja semantyki/wartości ogranicznika w treningu, tak jak randomizuje się
geometrię — żeby polityka nie mogła dostroić się do jednej konkretnej implementacji.

#### Nagrania

`docs/video/model_e5_sukcesy.mp4` (ziarna 1234/1248/1257, 222/178/109 kroków, 25.5 s) oraz
`docs/video/model_e5_typowy.mp4` (ziarno 1240, 320 kroków, **9 kroków kolizji — mediana**).
Oba w pełnej obwiedni (`--acc-limits`). Liczby kroków zgodne co do jednego z CSV ewaluacji.
**`watch_board` nie miał `--acc-limits`** — bez tej poprawki nagrania pokazywałyby ruchy,
których robot nie wykona (ta sama klasa błędu co brakujący `base_x_offset`, 6.42).

#### Stan na koniec kampanii

**Model: `logs/board_6dof/spot_e5/checkpoints/ppo_board_74123552_steps.zip`** — n=40, pełna
obwiednia sprzętowa (korpus Spota + kamera + limity prędkości **i przyspieszeń** + E-stop, h=2000):
**38/40 (95 %), 17.5 s, kolizje 20.9 kroku/ep (mediana 9), misuse 0.2, ‖Δa‖ 0.998, 0 zacięć.**
Alternatywa: koniec etapu 4 (66.12M) — 39/40, 18.8 s, kolizje 18.7, ale misuse 2.9. Różnice
nieistotne; wybrany e5 ze względu na **misuse 0.2 wobec 2.9**, czyli tryb grożący kamerze nadgarstkowej.

### 6.46. Kara za żądanie ponad limit i randomizacja obwiedni (2026-10-06/07)

**Punkt wyjścia (6.45).** Polityka nauczyła się *eksploatować* nasycenie ogranicznika:
po 16M kroków treningu wewnątrz obwiedni żądała przyspieszeń ponad limit w **74.8 %** kroków,
czyli częściej niż przed treningiem w obwiedni (62.0 %). Model e5: 38/40, 17.5 s,
kolizje 20.9 kroku/ep (mediana 9).

Sprawdzone dwa pomysły, każdy osobno i oba razem:

* **Wariant A — kara za żądanie.** Istniejące kary widziały tylko ruch **zrealizowany** (już
  przycięty), więc nadmiar żądania był dla nich niewidoczny. Nowy człon czyta `_acc_demand`,
  czyli wartość **przed** przycięciem — jedyne miejsce w pętli, gdzie ona istnieje
  (`envs/button_board.py`, hak limitów). Dwa składniki: `acc_demand_cost` karze nadmiar,
  `acc_margin_bonus` nagradza zapas — żeby gradient istniał także **tam, gdzie polityka jest
  już zgodna**, a nie tylko na pograniczu.
* **Wariant B — randomizacja obwiedni** (`joint_acc_limits_noise: [0.6, 1.4]`): limit losowany
  per epizod, żeby polityka nie mogła dostroić się do jednej konkretnej wartości i semantyki
  przycinania (wniosek z 6.45 o zerowym marginesie na różnicę wobec firmware).

**Kalibracja A z pomiaru.** Pierwsza wersja — kwadratowa i nieograniczona — dawała
**−274 nagrody na epizod** (274× nagroda zadania). Przyczyna: `_acc_demand` to maksimum po
podkrokach, a impulsy kontaktu z przyciskiem dają chwilowe żądania rzędu dziesiątek `g`.
Wersja finalna: liniowa i **ograniczona do 3× limitu**,
`nad = clip((żądanie − lim)/lim, 0, 3)`, `zapas = max(lim − żądanie, 0)/lim`.
Koszt 0.1 → ok. 0.27 nagrody epizodu, premia 0.2 → ok. 0.12. Premia **celowo mniejsza** od kary:
jest naliczana za krok, więc przy jej przewadze polityka opłacałoby się przedłużać epizody.

**Wyniki** (n=40, pełna obwiednia sprzętowa, h=2000, `--acc-limits`):

| bieg | sukcesy | czas | kolizje śr. (mediana) | misuse | ‖Δa‖ | żądania ponad limit |
|---|---|---|---|---|---|---|
| e5 — punkt wyjścia (74.12M) | 38/40 (95 %) | 17.5 s | 20.9 (9) | 0.2 | 0.998 | 74.8 % |
| **A — e6a (78.12M)** | **39/40 (98 %)** | **15.2 s** | **8.6 (4)** | 0.6 | 1.073 | **70.8 %** |
| A+B — e6ab (86.12M) | 40/40 (100 %) | 14.2 s | 5.2 | 0.1 | 1.039 | **78.1 %** |

Wariant B **sam** nie ma oceny: bieg zatrzymany na 12 % (`spot_e6b/ppo_board_74623552_steps.zip`)
decyzją o nieblokowaniu laptopa na kolejne godziny.

**Testy na parach.** Kolizyjność e6a wobec e5: **p = 0.047** — poprawa istotna, i to jedyna
istotna zmiana w całej kampanii po etapie 3. **⚠ Jej przypisanie karze zostało podważone
w 6.48:** kara liczy tętnienie kontrolera, nie zachowanie polityki, więc poprawa pochodzi albo
z samych 4M dodatkowych kroków, albo z działania tego członu jako niekierunkowego regularyzatora.
Nie wiadomo, i bieg kontrolny (te same 4M bez kary) nie został wykonany. Natomiast e6ab wobec e6a: **p = 0.75** —
dodanie randomizacji nie dało nic ponad samą karę, mimo ładniejszych liczb surowych
(40/40, kolizje 5.2). Lekcja z 6.45 o medianie obowiązuje: kolizyjność ma ciężki ogon.

**Kryterium nie zostało spełnione.** Zapisane przed biegami: spadek odsetka żądań ponad limit.
Faktycznie 74.8 → 70.8 → **78.1 %**, czyli A obniżył marginalnie, a A+B podniósł powyżej punktu
wyjścia. Wartość kampanii leży więc w kolizyjności (20.9 → 8.6), nie w tym, co było jej celem.
Dlaczego kara nie ruszyła tej metryki — patrz **6.47**, gdzie samo postawienie problemu okazało
się błędne.

#### Ogranicznik tempa komendy (`action_rate_limit`) — brak taniego obszaru

Skoro kara nie wymusza gładkości, sprawdzony ogranicznik **twardy**: przycięcie przyrostu
komendy między krokami, nałożone na gotowy model e6ab bez douczania (n=8, pełna obwiednia):

| limit | sukcesy | kolizje kroku/ep | ‖Δa‖ |
|---|---|---|---|
| brak | 8/8 | 3.1 | 1.037 |
| 0.50 | 8/8 | 7.9 | 1.043 |
| 0.25 | 7/8 | 19.2 | 0.988 |
| 0.10 | 6/8 | 62.9 | 0.719 |

Przy 0.50 **‖Δa‖ się nie zmienia** — limit jest per składowa, a norma jest zdominowana przez
sumę po sześciu osiach, więc taki limit jeszcze nie wiąże. Od 0.25 wiąże i natychmiast psuje:
kolizje rosną 6× i 20×, sukcesy spadają. **Nie ma obszaru, w którym ogranicznik dokłada
gładkość tanio.**

#### Filtr EMA po treningu — gładkości nie da się dołożyć po fakcie

Ten sam wniosek drugą metodą, na modelu 28.02M z 6.40 (`--action-ema`, n=40, obwiednia sprzętowa):

| α | sukcesy | kolizje kroku/ep | ‖Δa‖ | czas sukcesu |
|---|---|---|---|---|
| brak | 35/40 (88 %) | 2.0 | 1.344 | 24.8 s |
| 0.5 | 31/40 (78 %) | 12.9 | 0.609 | 40.9 s |
| 0.3 | 18/40 (45 %) | 111.4 | 0.340 | 52.4 s |
| 0.2 | 8/40 (20 %) | 95.0 | 0.247 | 64.7 s |

Filtr obniża ‖Δa‖ dokładnie tak, jak obiecuje — i równie dokładnie niszczy zadanie, bo polityka
jest **sprzężeniem zwrotnym**: wygładzona komenda nie dojeżdża tam, gdzie polityka ją wysłała,
a kolejna decyzja zapada już na podstawie rozjechanego stanu. **Gładkość musi wejść do treningu
(jak `action_rate_cost` w 6.43), nie do toru wykonawczego.** To ta sama klasa błędu co
„nałożymy limity dopiero na sprzęcie" z 6.31.

### 6.47. Sprostowanie: „żądanie ponad limit" nie było patologią. Sygnał nasycenia w obserwacji (2026-10-07)

**Trzy etapy ścigały nie-problem.** Metryka „odsetek kroków z żądaniem przyspieszenia ponad
limit" (62 % → 74.8 % → 70.8 % → 78.1 %) była w 6.44–6.46 traktowana jako patologia transferu.
To był **błąd w postawieniu problemu**, wskazany przez użytkownika:

> skoro ogranicznik jest w pętli, żądanie ponad limit **nic nie daje i nic nie kosztuje**.

Ruch po przycięciu jest **identyczny** niezależnie od tego, o ile polityka poprosiła powyżej
limitu. Ten wymiar wyjścia nie wpływa więc na nic, co polityka obserwuje lub za co dostaje
nagrodę — gradient jest w nim **dokładnie zerowy**, a wartość **dryfuje**. Wzrost 62 → 78 % nie
jest „nauczeniem się eksploatacji nasycenia" (jak zapisano w 6.45), tylko błądzeniem losowym po
kierunku, który nie ma zwrotu. I dlatego żadna kara go nie ruszyła: polityka nie „odmawiała"
zgodności — **nie miała czego optymalizować**. Zdanie z 6.45 o „polityce komenderującej mocniej,
bo i tak zostanie przycięta" jest nieuprawnioną interpretacją tej samej liczby.

**Ale wniosek jest odwrotny do porzucenia tematu.** Właśnie dlatego, że żądanie ponad limit nic
nie daje, **zgodność jest darmowa**: polityka komenderująca na limicie dostaje ten sam ruch co
komenderująca 3× powyżej, więc nie traci ani skuteczności, ani czasu. W odróżnieniu od gładkości
(6.43, cena: −23 % ‖Δa‖ za czas) i od samych limitów przyspieszeń (6.44, cena: 5.5 s → 17.5 s)
**tu nie ma kompromisu.** A zysk jest realny i nie jest kosmetyczny: jeśli polityka trzyma się
wewnątrz obwiedni, to **przestaje mieć znaczenie, jak firmware Kinovy przycina**, bo przycinanie
nigdy się nie włącza. To usuwa całą klasę ryzyka transferu zidentyfikowaną w 6.45 („zerowy
margines na różnicę semantyki przycinania"), a nie tylko ją mierzy.

**Czego brakowało: polityka nie widzi, że została przycięta.** W obserwacji ma prędkości stawów,
ale nie ma żadnego sygnału „twoja komenda nie przeszła". Kara z 6.46 wymagała od niej powiązania
własnego wyjścia z nagrodą przez samą korelację, bez przesłanki w wejściu. Stąd nowy sensor:

```python
@sensor(modality="object")
def joint_acc_saturation(obs_cache):   # 6 wymiarów
    nad = np.clip((self._acc_demand - lim) / lim, 0.0, 1.0)
```

0.0 = polityka mieści się w obwiedni; 1.0 = żądanie dwukrotnie ponad limit. Zerowy, gdy limit
przyspieszeń wyłączony — wtedy nie ma przycinania.

**Zmiana wymiaru obserwacji 120 → 132** (sensor wchodzi raz samodzielnie i raz w agregacie
`object-state`). Modele sprzed 2026-10-07 mają wejście 120 i **nie wczytają się**, dlatego
`obs_acc_saturation` jest **domyślnie wyłączone** — cała historia ewaluacji pozostaje wykonywalna.
To ta sama pułapka co modele 7DOF/6DOF (obs 128 vs 120).

**Wznowienie przez przeszczep wag.** Model e6a (obs 120) nie wczyta się do sieci o 132 wejściach,
więc zbudowany został `spot_e6a/ppo_board_78123552_obs132.zip`: te same wagi, kolumny pierwszej
warstwy odpowiadające nowym wejściom (47:53 i 113:119) **wyzerowane**. Polityka startuje
zachowaniem identycznym — zweryfikowane na losowych obserwacjach, **max |różnica akcji| = 3.0·10⁻⁷**.
Zmienną niezależną jest więc wyłącznie obecność sygnału; kary zostają na wartościach z wariantu A
(0.1 / 0.2), bez randomizacji (B pogorszył metrykę: 70.8 → 78.1 %).

**Kryterium, zapisane przed biegiem.** Spadek żądań ponad limit **przy zachowanej skuteczności** —
bo zgodność jest darmowa, więc utrata skuteczności znaczyłaby, że żądanie ponad limit **jednak coś
polityce daje**, czyli nasz ogranicznik nie jest neutralny. To byłoby odkrycie ważniejsze od samego
wyniku. Jeśli żądania nie drgną mimo sygnału w obserwacji, następnym krokiem jest **podkręcenie
kary wielokrotnie** — przy braku kompromisu nie ma czym ryzykować, a ostrożna kalibracja do 16 %
nagrody (6.45, 6.46) była nieuzasadniona i gubiła sygnał w szumie.

**Protokół pomiaru poprawiony.** Skrypt `zadanie_acc.py` mierzył żądanie przy ograniczniku
**wyłączonym** (żądanie = zrealizowany przyrost prędkości). Dla etapu 7 to nie przechodzi: przy
wyłączonym limicie sensor nasycenia jest stale zerowy, więc polityka dostawałaby wejście, którego
nie widziała w treningu. Nowy `zadanie_acc_on.py` mierzy przy **włączonym** ograniczniku, czytając
`_acc_demand` wprost — czyli dokładnie tę liczbę, którą karze `acc_demand_cost` i którą widzi
sensor. Protokół pasuje do warunków wdrożenia i jest ten sam dla obu modeli, ale **liczb z obu
skryptów nie wolno zestawiać** (74.8 % / 70.8 % / 78.1 % pochodzą ze starego protokołu).

#### Wynik: NEGATYWNY, bieg zatrzymany na 6.8M z 8M

Bieg (`ppo_board_6dof_spot_e7.yaml`, katalog `logs/board_6dof/spot_e7`) przebiegał stabilnie
(`train/std` 0.586 bez eksplozji, `approx_kl` 0.019–0.026, `explained_variance` 0.5–0.8),
ale karana wielkość nie drgnęła. Pomiar nowym protokołem (ogranicznik włączony, statystyka
**per epizod**, n = 8 ziaren):

| model | mediana żądania j1–j3 | `nad` (dokładnie to, co karze kara) | kroki na sufcie 3.0 |
|---|---|---|---|
| e5 — **bez kary** | 8.59 / 11.00 / 17.66 | **1.405** | 5.2 % |
| e6a | 7.18 / 13.99 / 19.57 | 1.720 | 16.3 % |
| e6ab | 9.92 / 14.22 / 18.54 | 1.528 | 7.6 % |
| e7 @1M | 10.31 / 13.43 / 20.17 | 1.617 | 10.4 % |
| e7 @3.2M | 8.64 / 11.16 / 15.54 | 1.398 | 6.4 % |
| e7 @6.5M | 6.50 / 12.09 / 18.93 | 1.503 | 7.4 % |

Model **bez kary** (e5) ma `nad` najniższe z całej stawki. Sygnał w obserwacji nie zmienił nic
po 6.5M kroków. **Przyczynę — i to, że cały pomiar był wadliwy — wyjaśnia 6.48.**

#### Dwie własne pomyłki pomiarowe po drodze, obie warte zapisania

**(1) Mediana po wspólnej puli kroków.** Pierwszy pomiar dał e6a 21.57/26.93/47.61 wobec
e7 5.94/11.84/18.54 i wyglądał na potwierdzenie hipotezy (spadek 3.6× przy zachowanej
skuteczności). Był artefaktem: oba te modele mają 7/8, czyli jeden epizod zacięty na 2000
krokach, co stanowi **ponad połowę wspólnej puli** — medianę wyznaczał więc ten jeden epizod
szarpania, nie polityka. Rozrzut median epizodowych j3 dla e6a: **13.9–112.9**, dla pozostałych
modeli 10–27. Po przejściu na statystykę per epizod wszystkie biegi wpadły w jedno pasmo.
To ta sama klasa błędu co „średnia zamiast mediany" z 6.45, tylko o poziom głębiej: tam ciężki
ogon był po epizodach, tu po krokach w obrębie puli. **Reguła: agregować najpierw w obrębie
epizodu, potem po epizodach — nigdy jedną medianą po wszystkim.**

**(2) „Kara siedziała na swoim sufcie".** Drugie wyjaśnienie, równie szybko postawione i równie
błędne: przy żądaniu 21 wobec limitu 5.2 `nad = clip((żąd−lim)/lim, 0, 3)` miało być stałą 3.0,
czyli znów zerowym gradientem. Pomiar: sufit wiąże w **5–16 %** kroków, nie w 100 %. Kara miała
gradient; nie działała z innego powodu (6.48).

### 6.48. Skala czasu, w której mierzymy przyspieszenie — i czego to NIE tłumaczy (2026-10-07)

**Obserwacja wyjściowa.** Przy `a = 0`, czyli komendzie „stój", ramię faktycznie stoi
(`|qvel|` mediana 0.02–0.09 rad/s, p99 ≤ 0.49), a mimo to `_acc_demand` raportuje:

| | j1 | j2 | j3 | j4 | j5 | j6 |
|---|---|---|---|---|---|---|
| **podkrok 2 ms** (`_acc_demand`: kara 6.46 i sensor 6.47) | 0.00 | 12.62 | 40.49 | 3.27 | **235.64** | 53.85 |
| **krok sterowania 50 ms** (`zadanie_acc.py`, 6.45) | 0.00 | 0.22 | 0.68 | 0.05 | **3.57** | 0.86 |

Różnica ~60×. `_acc_demand` nie mierzy „ile polityka żąda", tylko **ile fizyka zrobiłaby
w ciągu 2 ms przy momentach z OSC** — a to dla stawów o znikomej bezwładności jest ogromne
(j5 ma `armature 0`, więc resztkowy błąd kompensacji grawitacji daje setki rad/s²).
Dlatego `a = 0` dawało `nad` **1.797**, czyli **więcej niż każdy wytrenowany model**
(e5 1.405, e6a 1.720, e7 1.40–1.62). **Kara z wariantu A i sensor z etapu 7 karmią się
tętnieniem kontrolera, nie zachowaniem polityki** — stąd oba wyniki negatywne, i to
z przyczyny, która nie dałaby się naprawić podkręceniem kary.

#### Hipoteza o artefakcie kosztu limitu — OBALONA własnym testem

Postawiłem dalej idącą tezę: jeśli budżet 5.2 rad/s² jest przycinany co 2 ms, to zjada go
tętnienie, a na ruch makroskopowy zostaje resztka — czyli spowolnienie 5.5 s → 17.5 s z 6.44
byłoby artefaktem implementacji, a nie ceną ograniczenia sprzętowego. Test: ogranicznik
przepisany tak, by referencją było **własne poprzednie polecenie** (`joint_acc_limit_ref="cmd"`),
odporne na tętnienie. Model e3 (100/100 w 5.5 s, uczony bez limitu), te same ziarna, n = 8:

| obwiednia | sukcesy | czas | kroków 50 ms ponad limit | przysp. 50 ms mediana | p99 |
|---|---|---|---|---|---|
| brak | 8/8 | **5.4 s** | 63.2 % | 2.54/2.22/3.47/3.78/4.06/1.87 | ~20 |
| `qvel` (6.44) | 8/8 | **33.1 s** | 0.0 % | 1.78/3.00/3.70/4.04/3.35/3.19 | 5.20/5.20/5.20/10.0/10.0/10.0 |
| `cmd` (nowy) | 8/8 | **33.1 s** | 0.0 % | identycznie | identycznie |

**Wyniki identyczne co do cyfry**, bo w haku `v_prev` czytane jest **przed** `fn()`, czyli jest
to dokładnie nasza poprzednia komenda — ogranicznik był referowany do komendy od początku,
a mój wariant jest no-opem. Pozostaje w env jako udokumentowana opcja (domyślnie `"qvel"`).

**Wniosek: koszt limitu 5.2 rad/s² jest prawdziwy.** Dowód niezależny od implementacji:
p99 zrealizowanych przyspieszeń w skali 50 ms siedzi **dokładnie na limicie** (5.20/5.20/5.20/
10.0/10.0/10.0), a 0.0 % kroków go przekracza — budżet idzie w realny ruch, nie w szum.
Odwrotnie niż bez limitu, gdzie mediany makroskopowe (2.2–4.1) **mieszczą się** w 5.2,
a narusza dopiero ogon (p99 ≈ 20, 63 % kroków z co najmniej jednym stawem ponad).
Obraz „model przekracza limit w 81–88 % kroków" (6.31, 6.44) pochodził ze skali podkroku
i jest tą samą pomyłką co kara: **typowy ruch e3 mieści się w obwiedni, łamią ją szczyty.**

#### Czy ktoś miał ten problem — przegląd literatury

Szukane: (a) mierzenie przyspieszenia różnicą skończoną prędkości symulatora, (b) kara za
komendę, która i tak zostaje przycięta, (c) koszt twardych limitów prędkości/przyspieszeń
w RL dla manipulatorów.

1. **Dokumentacja MuJoCo, „Numerical integration"** — wprost o pomiarze:
   *„mjData.qacc holds the velocity difference (v+ − v)/h rather than the continuous-time
   acceleration"*, *„the same state yields a different qacc under a different timestep; as
   h → 0 it converges to the continuous acceleration"*, a dla jednokrokowych integratorów
   (Euler — **nasz**, bo `base.xml` nie ustawia `integrator`): *„finite-differenced accelerations
   (v_{t+h} − v_t)/h will not correspond to the continuous-time acceleration mjData.qacc"*.
   Czyli wielkość liczona tak, jak liczy ją nasz hak, **zależy od kroku całkowania** i nie jest
   przyspieszeniem fizycznym. To jest nasz błąd pomiarowy, nazwany w dokumentacji silnika.
   <https://mujoco.readthedocs.io/en/stable/computation/index.html> ·
   `armature` jako regularyzator bezwładności: <https://mujoco.readthedocs.io/en/stable/XMLreference.html>
2. **Fujita, Maeda, „Clipped Action Policy Gradient", ICML 2018** — dokładnie problem z 6.47:
   *„when policy gradient methods are applied to such tasks, out-of-bound actions need to be
   clipped before execution, while policies are usually optimized as if the actions are not
   clipped"*; agent *„receives the same reward for any sampled action that falls outside the
   bounds"*. To jest — w literaturze i od 2018 r. — ta sama obserwacja, którą dostałem jako
   korektę: komenda ponad limit nic nie daje i nic nie kosztuje, więc gradient jest zerowy.
   Praca podaje też wprost nasz wariant A jako znane obejście (*„penalize the norm of actions
   to prevent the policy from going too far out of the bounds"*) oraz własne rozwiązanie:
   estymator gradientu **świadomy przycinania** (nieobciążony, mniejsza wariancja).
   <https://arxiv.org/abs/1802.07564>
3. **robosuite — `LinearInterpolator` z `ramp_ratio`** (`robosuite/utils/traj_utils.py:25`,
   podłączany w `controllers/parts/arm/osc.py` jako `interpolator_pos`/`interpolator_ori`):
   `total_steps = ceil(ramp_ratio * controller_freq / policy_freq)`, domyślnie 0.2. To **własny
   mechanizm frameworka** do tego, czego próbujemy: rozciągnięcia celu OSC na krok sterowania,
   czyli ograniczenia przyspieszenia **zadanej trajektorii** przez konstrukcję, zamiast
   przycinania scałkowanej prędkości po fakcie. Dokumentacja wiąże go wprost z wykonalnością
   na sprzęcie. **Sprawdzony w 6.49:** do gotowego modelu dołożyć się nie da (0/8), ale dla
   stawów dużych obniża p99 z ~20 do 3.3–6.7 bez żadnego przycinania.
   <https://robosuite.ai/docs/simulation/controller.html>
4. **„Non-Prehensile Throwing: A Reinforcement Learning Perspective"** — ta sama architektura
   częstotliwości co u nas (polityka 20 Hz, symulacja 120 Hz, kontroler sprzętowy 500 Hz)
   i kary za **przyspieszenie i szarpnięcie** jako regularyzacja *„to keep the motion within
   the robot's physical limits"*. Potwierdza, że kara na przyspieszenie jest standardem — ale
   mierzona w skali polityki, nie podkroku. <https://www.alphaxiv.org/abs/2609.00771>
5. **Aljalbout, Frank, Karl, van der Smagt, „On the Role of the Action Space in Robot
   Manipulation Learning and Sim-to-Real Transfer"** (arXiv 2312.03673, 2023/24) — 250 agentów,
   13 przestrzeni sterowania, wnioski o tym, które przestrzenie dają polityki spełniające
   ograniczenia prędkości/przyspieszenia/szarpnięcia przy transferze. Nasza OSC_POSE jest
   jedną z porównywanych. <https://arxiv.org/abs/2312.03673>
6. **„Generating Realistic Arm Movements in Reinforcement Learning: A Quantitative Comparison
   of Reward Terms and Task Requirements"** (arXiv 2402.13949) — ilościowe porównanie członów
   nagrody wymuszających realistyczny, gładki ruch ramienia; przydatne do kalibracji wag
   zamiast strzelania (nasze 0.1/0.2 brane „na oko" dwukrotnie okazały się nietrafione).
   <https://arxiv.org/pdf/2402.13949>

**Czego NIE znalazłem:** opisu dokładnie naszej pomyłki, czyli kary i obserwacji liczonych
w skali podkroku fizyki przy polityce działającej 25× rzadziej. Najbliżej jest dokumentacja
MuJoCo (zależność różnicy skończonej od `h`) i CAPG (kara za przyciętą komendę).
Pierwszy wynik wyszukiwania sugerował opis tętnienia prędkości w MJX przy ciasnych limitach
długości cięgien — **sprawdziłem pełny tekst wskazanej pracy i tego tam nie ma**, więc nie
cytuję (snippet wyszukiwarki nie był pokryty treścią).

#### Co z tego wynika dla następnego kroku

Kary i sygnału nasycenia **nie da się naprawić przez wzmocnienie** — trzeba je przeliczyć na
skalę kroku sterowania (50 ms), bo tylko tam wielkość ma sens fizyczny. Tańszy kandydat niż
dalsze kary: `LinearInterpolator` z `ramp_ratio = 1.0` na celu OSC (punkt 3), który ogranicza
przyspieszenie zadanej trajektorii z konstrukcji — **sprawdzony w 6.49, wymaga treningu
od początku.** Punkt odniesienia do porównania: e3 bez limitu 5.4 s, e3 z limitem 33.1 s.

### 6.49. Interpolator OSC jako obwiednia zamiast przycinania — nie da się dołożyć po treningu (2026-10-07)

**Skąd pomysł.** Z przeglądu literatury (6.48, punkt 3): robosuite ma własny `LinearInterpolator`
(`utils/traj_utils.py:25`), podłączany do OSC przez `interpolation: "linear"` i `ramp_ratio`
w konfiguracji części `right`. Przy `controller_freq = 1/timestep = 500 Hz` i `policy_freq = 20 Hz`
daje `total_steps = ceil(ramp_ratio · 25)`, więc `ramp_ratio = 1.0` rozciąga cel OSC na **cały**
krok sterowania. Ogranicza to przyspieszenie **zadanej trajektorii z konstrukcji**, w odróżnieniu
od naszego haka, który przycina scałkowaną prędkość po fakcie. Dokumentacja robosuite wiąże ten
mechanizm wprost z wykonalnością tych samych komend na sprzęcie.

**Pomiar.** Model e3 (100/100 w 5.5 s, uczony **bez** interpolatora), te same ziarna, n = 8:

| wariant | sukcesy | czas | kroków 50 ms ponad limit | j1/j2/j3 mediana (p99) | j4/j5/j6 mediana (p99) |
|---|---|---|---|---|---|
| bez limitu, bez interp. | **8/8** | **5.4 s** | 63.2 % | 2.54/2.22/3.47 (20.8/20.0/21.2) | 3.78/4.06/1.87 (18.6/21.0/19.2) |
| bez limitu, `ramp 0.5` | **0/8** | 100.0 s | 78.0 % | 0.49/0.17/1.14 (4.8/3.5/6.8) | 16.90/6.32/17.37 (43.7/21.8/47.6) |
| bez limitu, `ramp 1.0` | **0/8** | 100.0 s | 55.2 % | 0.04/0.04/1.09 (**6.3/3.3/6.7**) | 8.71/3.67/6.17 (43.5/17.2/44.2) |
| limit, bez interp. | **8/8** | **33.1 s** | 0.0 % | 1.78/3.00/3.70 (5.2/5.2/5.2) | 4.04/3.35/3.19 (10.0/10.0/10.0) |
| limit + `ramp 1.0` | **0/8** | 100.0 s | 0.0 % | 0.00/0.01/1.30 (3.6/3.1/5.2) | 4.07/1.56/4.00 (10.0/9.9/10.0) |

Dwa punkty bez interpolatora wyszły **identycznie co do cyfry** jak w 6.48 (5.4 s i 33.1 s) —
pomiar jest powtarzalny.

**Wynik: interpolatora nie da się dołożyć do wytrenowanej polityki.** Oba `ramp_ratio` dają
**0/8** i pełny horyzont 2000 kroków. Mechanizm jest ten sam co przy dławieniu wzmocnień
w Webots (6.29) i przy filtrze EMA (6.46): cel osiągany dopiero po części kroku to efektywnie
słabsze wzmocnienie, a polityka jest sprzężeniem zwrotnym — kolejna decyzja zapada już na
podstawie stanu, do którego komenda nie dotarła. **Trzecie potwierdzenie tej samej reguły:
obwiednia i gładkość muszą wejść do treningu, nie do toru wykonawczego.**

**Czego pomiar NIE rozstrzyga.** Dla stawów dużych interpolator robi dokładnie to, co obiecuje:
przy `ramp 1.0` **bez żadnego przycinania** p99 j1–j3 spada z ~20 do 3.3–6.7, czyli praktycznie
w obwiednię 5.2. To jest obiecujące, ale **nie jest dowodem taniej obwiedni**, bo zmierzone
w reżimie porażki: przy 0/8 ramię szarpie się w miejscu, więc niskie przyspieszenia dużych stawów
mogą po prostu znaczyć, że te stawy stoją. Symetrycznie: wzrost nadgarstka (j4, j6 p99 ≈ 44
wobec ≈ 19) to artefakt szarpania, nie wniosek o interpolatorze. **Rozdzielenie wymaga treningu
z interpolatorem** — inaczej mierzymy niezgodność trening/test, nie obwiednię.

**Pytanie otwarte i jego koszt.** „Czy interpolator jest tańszą obwiednią niż clip" wymaga biegu
8M kroków z `interpolation: "linear"` od początku (ok. 8–9 h zegara). Punkty odniesienia do
porównania są gotowe: **bez limitu 5.4 s, z clipem 33.1 s**, oba przy 8/8.

**Zastrzeżenie pomiarowe.** Kolumna kolizji w tym biegu jest **nieważna** — skrypt czytał
`info["collision"]`, którego env nie zwraca (kolizja jest dostępna tylko metodą
`_check_board_collision()`, tak jak czyta ją `compare_models.py`). Skrypt poprawiony,
liczby kolizji trzeba zmierzyć ponownie, gdy wariant będzie tego wart.

### 6.50. Wybór modelu na sprzęt przy n=100 i wierność semantyki przycinania (2026-10-07)

#### Wybór modelu: n=100 dla e6a i e6ab

Przy n=40 różnica między wariantem A (e6a) i A+B (e6ab) była nieistotna (p = 0.75), więc decyzji
o modelu na sprzęt nie było na czym oprzeć. Powtórzone przy n=100, pełna obwiednia sprzętowa
(korpus Spota + kamera + limity prędkości i przyspieszeń + E-stop, h = 2000, `--acc-mode clip`):

| | e6a (78.12M) | **e6ab (86.12M)** |
|---|---|---|
| sukcesy | 98/100, **[93.0–99.4 %]** | **100/100, [96.3–100 %]** |
| czas sukcesu | 15.6 s | 15.2 s |
| kolizje kroku/ep (mediana) | 8.7 (4.5) | 7.0 (6.0) |
| misuse kroku/ep | 0.7 | **0.0** |
| kontakt ze Spotem | 0.2 | 0.1 |
| ‖Δa‖ | 1.035 | 1.048 |
| każdy przycisk | 100/99/98/99/100 | **100/100/100/100/100** |
| zacięcia | 0 | **0** |

**Testy na parach (n=100, te same ziarna): żadna miara nie różni istotnie.** Sukcesy — tylko
2 pary niezgodne, obie na korzyść e6ab, p = 0.50. Kolizje 51:46, p = 0.92. Czas 52:48, p = 0.82.
Misuse 1:5 na korzyść e6ab, p = 0.093 (granicznie).

**Wybrany `spot_e6ab/checkpoints/ppo_board_86123552_steps.zip`** — nie dlatego, że jest istotnie
lepszy, bo nie jest, ale dlatego, że przy równości wszędzie **nie ma ani jednego trybu porażki**:
100/100, każdy przycisk 100/100, misuse 0.0, 0 zacięć. Kryterium to samo, co przy wyborze e5
nad e4 (6.44): misuse jest trybem grożącym kamerze nadgarstkowej, więc przy nierozstrzygalnej
reszcie decyduje on.

**Uwaga do 6.46.** Tam e6ab wobec e6a wyszło nieistotnie (p = 0.75) i tak zostaje — n=100 tego
nie zmieniło, tylko potwierdziło **oba** modele (e6a 39/40 → 98/100, e6ab 40/40 → 100/100).
Tym razem lekcja z 6.40 („najlepszy przy n=40 był najgorszy przy n=100") **się nie powtórzyła**.

#### Semantyka przycinania przyspieszeń: nasza była wygodniejsza dla polityki, niż firmware

Pytanie postawione przed sprzętem nie brzmi „czy polityka przekracza limity" (nie przekracza:
0.0 % kroków 50 ms ponad limitem, patrz 6.48), ale **„czy wytrzyma inną semantykę przycinania"**.
Nasz hak przycinał **niezależnie na każdym stawie**, co zmienia wypadkowy kierunek ruchu: staw
na limicie zostaje ścięty, pozostałe nie. Instrukcja Gen3 R07 opisuje dla toru twist zachowanie
inne — **proporcjonalne skalowanie z zachowaniem kierunku**. Dodany `joint_acc_limit_mode`
(`"clip"` / `"scale"`) i flaga `--acc-mode` w `compare_models.py`. Obie wersje trzymają **tę samą
obwiednię**; różnica jest wyłącznie w kierunku ruchu po przycięciu.

e6ab, n=40, te same ziarna, pary:

| miara | `clip` (nasze) | `scale` (jak firmware) | pary gorzej:lepiej | p |
|---|---|---|---|---|
| sukcesy | 40/40 | **40/40** | — | — |
| czas | 284 kr. | 293 kr. | 17:23 | 0.83 |
| **kolizje** | śr. 5.2, med. 5.0 | **śr. 11.9, med. 6.5** | **22:13** | **0.033** |
| misuse | 0.1 | 0.0 | 0:1 | 0.32 |

**Zadanie i czas przenoszą się bez zmian, kolizyjność rośnie istotnie** — 2.3× w średniej,
+30 % w medianie (podane obie, bo rozkład ma ciężki ogon, 6.45). Wniosek praktyczny: **nasza
ewaluacja zaniżała kolizyjność oczekiwaną na sprzęcie.** Liczba, którą należy cytować dla
sprzętu, to **11.9 kroku kolizji/epizod, nie 5.2** — nadal znacznie poniżej e5 (20.9).

**Niespójność, którą to odkryło.** Limit **prędkości** działał u nas od 6.37 w trybie `scale`,
a limit **przyspieszeń** w `clip`. Nie było to wyborem, tylko przeoczeniem. Wierniejszy jest
`scale` i tak powinny iść przyszłe treningi oraz oceny na sprzęt. Domyślna wartość w env
zostaje `"clip"` **świadomie** — przestawienie jej zmieniłoby znaczenie każdego wcześniejszego
skryptu oceny bez śladu w logach. Decyzja o zmianie domyślnej należy do użytkownika.

**Czego brakuje:** n=100 w trybie `scale` (mamy tylko n=40) oraz treningu w tym trybie — model
uczony przy `clip` pracuje na nasyceniu, więc może odzyskać część kolizyjności, jeśli nauczy się
w semantyce docelowej.

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

### 7.9. Conda przykrywa Pythona ROS-a (Webots/ROS 2)

`ros2 launch` kończył się pętlą: `Ros2Supervisor` padał, `reset_handler`
respawnował węzły, nowy driver dostawał `refusing connection attempt from another
extern controller`, i tak w kółko (78 osieroconych procesów z jednego wybuchu).
Przyczyna w tracebacku supervisora:

```
File "/home/.../miniconda3/lib/python3.11/importlib/__init__.py", line 126
ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'
The C extension '..._rclpy_pybind11.cpython-311-...so' isn't present
```

ROS Humble jest zbudowany pod Pythona 3.10, a aktywne środowisko `conda base`
podstawia 3.11 (`miniconda3/bin` na początku `PATH`). Drivery `Gen3` i `Spot` to
binarki C++ i łączyły się poprawnie — padał wyłącznie `ros2_supervisor.py`, co
maskowało przyczynę. Rozwiązanie: `conda config --set auto_activate_base false`,
do robosuite `conda activate robosuite` jawnie. **Trwałe ograniczenie workflow:
robosuite wymaga condy, ROS wymaga jej nieobecności — dwa osobne terminale.**

### 7.10. Stary workspace przykrywa pakiety z apta

`~/.bashrc` sourcował `ros2_ws/install/setup.bash`, gdzie leżał
`webots_ros2_driver` zbudowany ze źródeł (era R2023b) **oraz druga kopia
`webots_ros2_spot` na gałęzi `main`, bez Kinovy**. Overlay wygrywał z wersją
2025.0.0 z apta. Objawy były trzy, pozornie niezwiązane:

1. `refusing connection` (patrz 7.9 — kaskada po padnięciu supervisora),
2. `AttributeError: 'NoneType' object has no attribute 'getSFVec3f'` w
   `spot_driver.py:153` — `getFromDef("Spot")` zwracało `None`, mimo że
   `DEF Spot Spot` jest w świecie (linia 575, `supervisor TRUE`). To była
   niezgodność API między driverem z R2023b a Webotsem R2025a,
3. ryzyko uruchomienia `spot_launch.py` z gałęzi `main`, czyli świata bez Gen3.

Po wyłączeniu tej linii wszystkie trzy zniknęły bez dotykania kodu. Wniosek
ogólny: **overlay ROS-owy z porzuconego workspace'a jest cichym źródłem awarii,
które wyglądają na błędy świata albo modelu.**

Pozostałe drobiazgi z tego samego uruchomienia:

- `webots_spot_msgs` nie było zbudowane w workspace projektu, bo `colcon build`
  uruchomiono **wewnątrz katalogu pakietu** (brak `src/`), więc zbudował się sam
  `webots_spot`. Trzy zależności deklaruje `webots_ros2_spot.repos`.
- `ros2 node list` zwracało pustkę mimo działających węzłów — nieświeży
  `ros2 daemon`, wystartowany pod starym środowiskiem. `ros2 daemon stop` albo
  `--no-daemon`.
- Ostrzeżenie `Contact joints ... only the 10 deepest contact points instead of
  all the 66` powtarza się co krok fizyki, bo `WorldInfo` nie ma **żadnego**
  węzła `ContactProperties`, więc obowiązuje domyślny wpis z
  `maxContactJoints 10`. Nieszkodliwe przy pozie spoczynkowej, ale istotne przy
  docisku płaskiej wkładki chwytaka do kapturka. MuJoCo nie ma takiego obcięcia
  i modeluje kontakt inaczej (`solref`/`solimp` zamiast złączy ODE) — **kontakt
  w obu symulatorach nie będzie identyczny i nie da się tego wystroić do
  zgodności**; to realna luka sim2sim, do nazwania wprost w pracy.
- Tekstury o wymiarach niebędących potęgą dwójki Webots przeskalowuje
  (`landolt.png` 92×92 → 128×128). Znaczniki ArUco robić w 256×256 lub 512×512,
  bo przeskalowanie rozmywa krawędzie, a ArUco żyje z ostrych krawędzi.
- `end_effector_link` w PROTO Gen3 ma węzeł `Physics` bez `boundingObject`, więc
  Webots wstawia **1 kg z jednostkową macierzą bezwładności** na bezmasową ramkę
  TF. Do usunięcia (ramka ma zostać ramką), bo zmienia dynamikę nadgarstka.


## 8. Narzędzia pomocnicze

- `training/watch_board.py` — podgląd polityki: okno na żywo albo `--record plik.mp4`
  (render offscreen EGL + ffmpeg, działa bez ekranu/przez SSH); auto-wybór modelu
  (best_model → najnowszy checkpoint), `--random`, `--episodes N`,
  `--robot Kinova3_6DOF`, `--activation-depth`, `--horizon` (modele 6DOF etapu 1
  wymagają 2000 — domyślne 1000 ucięłoby epizod), `--seed` (parowanie jak
  w `compare_models.py`: epizod *k* dostaje `seed+k`, więc można odtworzyć
  konkretny epizod z CSV ewaluacji — zweryfikowane co do kroku).
- `training/compare_models.py` — ewaluacja/porównanie na parowanych epizodach;
  `--robot`, `--activation-depth`, `--horizon`, `--free-gripper`, `--csv`,
  test Fishera dla dwóch modeli, rozbicie **wciśnięć per przycisk** (bez niego
  problem btn3 jest niewidoczny w zbiorczym success rate). **`--free-gripper`
  jest obowiązkowe dla modeli uczonych z wolnym chwytakiem** (wszystkie biegi
  6DOF od `v4like` wzwyż) — bez niego pomiar zaniża wynik kilkukrotnie
  (rozdz. 6.16).
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
5. ~~Retrening 6DOF od zera na poprawionej geometrii~~ — **zrobione**
   (rozdz. 6.16, 6.17). 6DOF osiąga **80/100** przy nominalnym progu 12 mm.
   Wyniki 6.8–6.14 pozostają nieaktualne: dotyczyły zadania z czterema
   usterkami geometrii i chwytaka.
   ~~**Kolejny etap = zacieśnianie** fine-tunem na gotowej polityce~~ —
   **zrobione** (rozdz. 6.18): zamknięty chwytak + bramka palcowa + pełne kary
   v2.3 dały **52/100** przy kolizyjności **4.7× niższej** (8.8 zamiast 41.6
   kroku/ep). Przewidywany koszt zamknięcia chwytaka (~32 pp) się potwierdził.
   Bieg **kolapsuje po ~2.5M kroków** — do powtórzenia z wygaszaniem
   `depress_shaping_k` albo `collision_cost = 2.0`, budżet 3M (punkt 11).
6. Powtórzyć na poprawionej geometrii pomiary, które opierały się na starej:
   skan `base_x_offset` (dawne „0.50 m od ściany = optimum"), porównanie
   manipulowalności 6DOF vs 7DOF, oraz baseline 7DOF przy **wolnym** chwytaku
   — bez tego porównanie 6DOF/7DOF pozostaje niekontrolowane.
7. Rozważyć tryb montażu „na podłodze" z regulaminu jako drugi wariant
   w domain randomization.
8. (Opcjonalnie) porównanie AprilTag vs YOLO jako rozdział pracy.
9. ~~**Obniżyć kolizyjność 6DOF**~~ — **zrobione rewardem** (rozdz. 6.18):
   `collision_cost` 0.5 → 5.0 zbiło kolizyjność 41.6 → 8.8 kroku/ep. To jedyny
   przypadek w projekcie, w którym naprawa zachowania rewardem się udała —
   w odróżnieniu od prób z rozdz. 6.5–6.6 chodziło o kontakt **ramienia**
   z tablicą, a nie o gracji ruchu, i kara nie dotykała kontaktu chwytaka
   z celem. Limit siły/prędkości TCP w `SafetyLayer` pozostaje potrzebny jako
   druga warstwa (pozostałe 8.8 kroku/ep + artefakt QACC z 6.18).
10. Wpiąć model 6DOF w nadzorcę sim2real zamiast modelu 7DOF. Kandydaci:
    `logs/board_6dof/s2/best_model/best_model.zip` (**74/100 przy h = 1500**,
    zamknięty chwytak + bramka palcowa = **protokół sprzętowy**, rekomendowany)
    albo `logs/board_6dof/depress_ft/ppo_board_final.zip` (80/100 przy h = 1000,
    ale wolny chwytak i 41.6 kroku kolizji/ep). Uwaga: **obs 120 vs 128**,
    modele nie są przenośne, `SimIO` wymaga przełącznika robota
    (`training/eval_supervisor.py` tworzy dziś na sztywno `robot="Kinova3"`).
    Test akceptacyjny nadzorcy powtórzyć przy h = 1500, nie 1000.
11. ~~**Powtórzyć etap zacieśniania bez kolapsu** przez obcięcie nagrody~~ —
    **zrobione, WYNIK NEGATYWNY** (rozdz. 6.20): bieg s3 zdegradował się tak
    samo jak s2, tylko szybciej, a model ze szczytu jest nierozróżnialny od
    punktu startowego (p = 1) przy **wyższej** kolizyjności. Hipoteza
    o konflikcie shaping-vs-kara obalona. Sufit ~52 % przy pełnym v2.3 jest
    odporny na strojenie rewardu — dalsze próby tą drogą **nie mają
    uzasadnienia** (to już trzecia u 6DOF i szósta w projekcie).
13. **Kolejny krok ma być STRUKTURALNY, nie rewardowy.** Kandydaci:
    ~~(a) dłuższy horyzont~~ — **zrobione i WYCZERPANE** (rozdz. 6.21, 6.23):
    51 → 74 → **85/100** przy h = 1000 / 1500 / 2000. Dalsze wydłużanie nie ma
    sensu **zadaniowego** (2000 kroków = 100 s wobec 75 s nadzorcy z retry),
    choć rozkład formalnie wciąż dotyka limitu (max 1926);
    (b) inna poza bazy / `base_x_offset` — skan nigdy nie został powtórzony
    na poprawionej geometrii (patrz punkt 6); (c) parametry OSC (sztywność,
    limit prędkości), które wpływają na precyzję docisku bez dotykania
    rewardu. **(b) i (c) pozostają nietestowane.**
14. **Rozważyć zamknięcie tematu na obecnym modelu.** `s2/best_model` =
    **74/100 przy h = 1500** (85/100 przy h = 2000; 51/100 przy nieaktualnym
    już h = 1000) przy zamkniętym chwytaku, bramce palcowej i pełnych karach
    jest modelem sprzętowo poprawnym; nadzorca ma warstwę retry, która u 7DOF
    podniosła 90/100 → 92/100 przy 75 s (rozdz. 5.1). Do pracy inżynierskiej
    wynikiem jest **udokumentowana ścieżka i uczciwie zmierzone ograniczenie**,
    a nie koniecznie wyższy procent.
12. Zmierzyć od nowa modele z 6.16–6.18 w środowisku ze ścianą (6.19).
    **Zrobione dla `s2/best_model`**: 51/100 przy 15.4 kroku kolizji/ep
    (sukces bez zmian, kolizyjność +75 % — pomiar w rozdz. 6.19). Pozostaje
    `depress_ft/ppo_board_final.zip` (80/100 bez ściany), jeśli będzie
    potrzebny do porównań w pracy.

15. **Geometria przeniesiona na wymiary z Webots (2026-09-01, rozdz. 6.24).**
    `BOARD_Z` 0.60 → 0.64, `KINOVA_BASE_Z` 0.65 → 0.62, promień gniazd
    0.120 → 0.107, szum percepcji ~3× w górę, nowa oś `base_z_noise = 0.03`.
    Do zrobienia w kolejności:
    - fine-tune 2M z `s2/best_model`
      (`training/configs/ppo_board_6dof_webots_ft.yaml`); kryterium: jeśli
      w ciągu 1M sukces nie wróci powyżej ~20 %, potrzebny scratch;
    - zmierzyć `s2/best_model` w NOWYM env **przed** startem fine-tunu, żeby
      mieć uczciwe „przed dostrojeniem" (bez tego nie ma z czym porównać);
    - sprawdzić rozbicie per przycisk — przy `base_z_noise = 0.03` skrajny
      przycisk potrafi leżeć 21 cm powyżej bazy i osiągalność nie jest
      zweryfikowana; kandydat na zawężenie do 0.02;
    - **zmierzyć miarką realną tablicę na uczelni** (wysokość montażu, promień
      gniazd, wysunięcie i skok grzybka). To rozstrzyga spór Webots↔robosuite
      zamiast go arbitrażować i jest samodzielnym wkładem do pracy:
      „geometria zmierzona" zamiast „geometria przyjęta".

16. **Przyciski w Webots nie mają fizyki** (rozdz. 6.24) — do zbudowania na
    każdym z pięciu gniazd: `Solid` z `boundingObject` i `physics`,
    `SliderJoint` ze sprężyną i tłumieniem, `PositionSensor` do odczytu
    głębokości. Odpowiednik naszego `<joint type="slide">` z `button.xml`.
    Dopiero po pomiarze realnego przycisku, żeby nie odwzorowywać wymiarów
    równie niepotwierdzonych jak obecne.

17. **Model na sprzęt zmieniony (2026-09-07, rozdz. 6.25).** Zamiast
    `s2/best_model` (53/100 @h=1500) obowiązuje
    `logs/board_6dof/sweep/ft_sc1_v23_long/ppo_board_final.zip` —
    **70/100 @h=1500** (77/100 @h=2000), średni sukces 24.8 s zamiast 44.6 s,
    kroki kolizji 6.6 zamiast 37.6, gładkość 1.229 zamiast 2.884. Przyczyną
    poprawy jest `log_std_max` (jeden callback), nie zmiana nagrody ani
    geometrii. Do zrobienia:
    - przemierzyć nadzorcę (rozdz. 5.1) na nowym modelu — retry przy 75 s
      powinien dać więcej niż wcześniejsze 92/100, bo epizody są 1.8x krótsze;
    - zbadać `stuck` 1 → 6: obejrzeć te epizody przez `watch_board --record`
      i sprawdzić, czy to jeden powtarzalny tryb awarii;
    - **przetrenować 7DOF z `log_std_max`** dla porównania — dotychczasowy
      baseline 7DOF jest z innej geometrii (6.22) i z polityką o `std` 13.7,
      więc porównanie 6DOF/7DOF nadal jest niekontrolowane;
    - rozważyć `log_std_max` jako domyślną wartość w `train_board.py`, a nie
      opcję — żaden bieg bez niego nie uniknął eksplozji.

18. **Nadzorca: wyzwalacz retry oparty na limicie `j4` (2026-09-07, rozdz. 6.25).**
    Diagnoza `stuck` dała mierzalny sygnał, którego nadzorca dziś nie używa:
    `j4` w granicach ok. 3° od limitu (`[-2.10, +2.10]` rad) przez N kolejnych
    kroków. Utknięcia spędzają tak 73.6 % epizodu, sukcesy 20 % (p = 0.0031).
    `joint_states` Kinovy podaje `j4` wprost, więc detektor jest identyczny
    w symulacji i na sprzęcie. Do zrobienia:

    - **Dodać `wrist_limit` jako drugi wyzwalacz w `RetryManager`** obok
      istniejącego `timeout=10 s`. Obecny wyzwalacz czeka 200 kroków bezczynnie;
      detektor `j4` wykryje blokadę w kilkadziesiąt kroków, oszczędzając budżet
      nadzorcy.
    - **Zmienić strategię odjazdu.** `RetryManager.update()` ustawia waypoint na
      `btn_top - press_axis * retreat_dist`, czyli **prosto w tył wzdłuż osi
      wciskania**. Ramię wraca tą samą linią i najpewniej ląduje w tej samej
      konfiguracji, więc `j4` znów oprze się o limit. Retry musi **zmieniać
      konfigurację**, nie tylko odsuwać TCP: odjazd w bok albo przez pozycję
      pośrednią o innym znaku `j4`.
    - **Zweryfikować zysk.** 9 z 24 porażek to utknięcia, więc pułap idealnego
      retry to **~85/100**, nie 100/100. Zmierzyć `supervisor/` na nowym modelu
      i porównać z 76/100 bez nadzorcy.
    - **Sim2real:** `percep_pos_bias` jest w symulacji stały na epizod, więc
      retry go nie zmienia. Na sprzęcie ArUco przeliczy pozę po odjeździe, ale
      tylko przy **zmianie punktu obserwacji** — retry z tej samej pozy kamery
      odtworzy ten sam błąd. Odjazd powinien więc zmieniać także widok kamery
      nadgarstkowej.
    - **Rozważyć naprawę przyczyny, nie skutku:** sprawdzić, czy przy montażu
      Gen3 na Spocie da się obrócić bazę ramienia tak, żeby `j4` pracował dalej
      od limitu w typowej pozie roboczej. To usunęłoby blokadę zamiast ją
      obchodzić.

19. **Sim2sim warstwa B2 — zamknięcie pętli (rozdz. 6.27).** Warstwy A, B0 i B1
    są domknięte: kontroler, obserwacja i polityka odtwarzają robosuite.
    Do zrobienia w kolejności:
    - `action_to_target()` — pozycja `eef_pos + action[0:3] * 0.05`, orientacja
      na razie bez zmian. **Odjąć `GRIP_OFFSET` wzdłuż lokalnej osi Z**:
      `read_tcp()` zwraca punkt chwytu, a `/compute_ik` oczekuje kołnierza,
      więc bez tego ramię celuje 14.5 cm za daleko. Delty orientacji **nie
      składać przez kąty Eulera** — to nie jest złożenie obrotów; poprawnie
      robi się to na kwaternionach, po domknięciu pozycji.
    - **Przeliczyć rzeczywistą pozę omniboardu z Webotsa do ramki bazy.**
      Most podaje dziś geometrię treningową (`button_pose()` zwraca stałe
      z robosuite). Kształt płyty w obu symulatorach jest zgodny, nieznana jest
      poza bazy względem niej. Bez tego ramię minie przycisk o kilka centymetrów
      i będzie to wyglądać na błąd polityki.
    - `pick_target()` — dziś stałe 0. Wykrycia wciśnięcia w Webots **nie da się
      zrobić**, bo przyciski nie mają fizyki (punkt 16), więc przełączanie celu
      musi na razie iść z nadzorcy po czasie albo po dystansie.
    - **Wymienić chwytak w Webots z 2f_140 na 2f_85** przed jakimikolwiek
      pomiarami kontaktu — w laboratorium jest Robotiq 85.
    - Odczytać rzeczywisty stan chwytaka zamiast stałych, gdy tylko sprzęt na to
      pozwoli. Polityka reaguje na ten blok nieproporcjonalnie mocno
      (rozdz. 6.27).

20. **Sim2sim B2 — sufit regulatora (rozdz. 6.29).** Obie ścieżki wykonawcze
    kończą na `/kinova_joint_trajectory_controller/joint_trajectory`, który
    realizuje ok. 1/3 zadanego punktu. To wąskie gardło, nie IK i nie Servo.
    Do zrobienia, w tej kolejności:
    - **Sterownik prędkościowy lub momentowy w Webots** zamiast położeniowego.
      To usuwa sufit u źródła, ale wymaga ingerencji w driver MASKOR.
    - **Fine-tune z randomizacją wzmocnień w robosuite** — opakowanie skalujące
      osie akcji przed OSC, współczynniki losowane na epizod (0.3–1.2).
      NIE w Webots: nie ma tam fizyki przycisków, więc nagroda jest
      nieobliczalna, a 1M kroków przy 20 Hz to ~14 h zegara.
    - Przeliczyć realną pozę omniboardu (punkt 19) — nadal nierobione, most
      celuje w geometrię treningową.
    - Zmienione w repozytorium MASKOR: `moveit_joint_limits.yaml`,
      `max_acceleration` 0.7→10.0 (stawy 1–3) i 1.5→15.0 (4–6). Oryginał
      zapisany w nagłówku pliku.

## 10. Źródła

Wszystko, na co powołują się liczby w tym dokumencie, z zaznaczeniem **gdzie dokładnie**
w źródle i **do czego** zostało użyte. Kolejność: sprzęt, oprogramowanie, literatura.
Przy każdym źródle rozdział, w którym jest wykorzystane.

### 10.1. Dokumentacja sprzętu

| źródło | co z niego wzięte | gdzie użyte |
|---|---|---|
| **Kinova, *User Guide Gen3 R07***, <https://www.kinovarobotics.com/uploads/User-Guide-Gen3-R07.pdf> — tab. 46 (limity ogólne), tab. 48 i 49 (przyspieszenia, sekcja 6 DoF), tab. 50 (momenty) | prędkości 1.39 / 1.22 rad/s; przyspieszenia **twarde** 5.2 (j1–3) i 10.0 (j4–6) rad/s²; rozróżnienie: 1.0 rad/s² z tab. 49 dotyczy **tylko** angular joystick / joint trajectory, nie toru twist; momenty 39 / 9 N·m; „acceleration will max out at acceleration limit" (robot nie faultuje) | 6.37, 6.44, 6.48 |
| **`ros2_kortex`**, `kortex_description/arms/gen3/6dof/config/joint_limits.yaml` oraz `twist_limits.yaml` | 0.8727 rad/s dla wszystkich 6 stawów (= 50°/s, limit admitancji); twist 0.5 m/s i 1.74 rad/s | 6.31, 6.32, 6.37 |
| **Karta katalogowa E-stopu** (grzybek Ø 22/30 mm, parametry producenta) | skok **5 mm**, rozwarcie styku 2 mm, siła **25 N**, zatrzask przy pełnym skoku | 4.3, 6.33 |
| **Robotiq 2F-85** (chwytak laboratoryjny, zgodny z symulacją; w Webots jest 2F-140) | zgodność modelu chwytaka; stałe `GRIPPER_*_MED` w moście | 6.27, 6.41 |

### 10.2. Oprogramowanie i jego dokumentacja

| źródło | co z niego wzięte | gdzie użyte |
|---|---|---|
| **MuJoCo, *Computation / Numerical integration***, <https://mujoco.readthedocs.io/en/stable/computation/index.html> | „mjData.qacc holds the velocity difference (v+ − v)/h rather than the continuous-time acceleration"; „the same state yields a different qacc under a different timestep"; dla integratorów jednokrokowych (Euler — nasz) „finite-differenced accelerations will not correspond to the continuous-time acceleration" | **6.48** — nazywa nasz błąd pomiarowy |
| **MuJoCo, *XML reference***, <https://mujoco.readthedocs.io/en/stable/XMLreference.html> | `armature` jako dodatkowa bezwładność stawu (nasz j5 ma 0, stąd setki rad/s² z resztkowego momentu); `solref`, `jnt_range` | 6.34, 6.48 |
| **robosuite — `LinearInterpolator`**, `robosuite/utils/traj_utils.py:25`, podłączany w `controllers/parts/arm/osc.py` jako `interpolator_pos` / `interpolator_ori`; <https://robosuite.ai/docs/simulation/controller.html> | `total_steps = ceil(ramp_ratio · controller_freq / policy_freq)`, domyślnie 0.2; własny mechanizm frameworka do ograniczania przyspieszenia **zadanej** trajektorii, wiązany w dokumentacji z wykonalnością na sprzęcie | 6.48, **6.49** — po treningu dołożyć się nie da (0/8); dla j1–j3 obniża p99 z ~20 do 3.3–6.7 |
| **MASKOR / FH Aachen, `webots_ros2_spot`**, <https://github.com/MASKOR/webots_ros2_spot>, gałąź `kinova_gen3` — `protos/Spot.proto`, `protos/Kinect.proto`, `resource/moveit_servo_config.yaml`, `resource/moveit_joint_limits.yaml` | geometria korpusu Spota i kamery przeliczona do ramki ramienia; TF `base_link → gen3_base_link` = [0.256, 0, 0.112]; `publish_period`; limity planera | 6.24, 6.29, 6.41, 6.42 |
| **`goldengeko/robosuite_training_examples`**, <https://github.com/goldengeko/robosuite_training_examples> | punkt wyjścia konfiguracji treningu | 6.7 |

### 10.3. Literatura

| źródło | co z niego wzięte | gdzie użyte |
|---|---|---|
| **Fujita, Maeda, *Clipped Action Policy Gradient*, ICML 2018**, <https://arxiv.org/abs/1802.07564> | „policies are usually optimized as if the actions are not clipped"; agent „receives the same reward for any sampled action that falls outside the bounds"; kara za normę akcji poza zakresem jako znane obejście; estymator gradientu świadomy przycinania | **6.47, 6.48** — dokładnie nasz problem zerowego gradientu przy przycinaniu, opisany od 2018 |
| **Aljalbout, Frank, Karl, van der Smagt, *On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer*** (arXiv 2312.03673), <https://arxiv.org/abs/2312.03673> | 250 agentów, 13 przestrzeni sterowania (w tym OSC) i ich wpływ na spełnianie ograniczeń przy transferze | 6.48 |
| ***Non-Prehensile Throwing: A Reinforcement Learning Perspective***, <https://www.alphaxiv.org/abs/2609.00771> | ta sama architektura częstotliwości (polityka 20 Hz, kontroler sprzętowy 500 Hz); kary za przyspieszenie i szarpnięcie „to keep the motion within the robot's physical limits", mierzone w skali polityki | 6.48 |
| ***Generating Realistic Arm Movements in Reinforcement Learning: A Quantitative Comparison of Reward Terms and Task Requirements***, <https://arxiv.org/pdf/2402.13949> | ilościowe porównanie członów nagrody wymuszających gładki ruch — do kalibracji wag zamiast dobierania „na oko" | 6.48 |

### 10.4. Źródła odrzucone

- **arXiv 2606.05236** — wynik wyszukiwania sugerował opis tętnienia prędkości stawów w MJX
  przy ciasnych limitach długości cięgien (i nieskuteczność filtru średniej kroczącej), co
  pasowałoby do 6.48. **Sprawdzony pełny tekst pracy: tego tam nie ma.** Snippet wyszukiwarki
  nie był pokryty treścią. Nie cytować.
- **PR #241 w `ros2_kortex`** (1.3963 / 1.2218 rad/s) — zmieniał tagi `<velocity>`
  w `gen3_macro.xacro`, **nie** plik limitów. Pierwotne założenie 1.6× za luźne, sprostowane
  w 6.31.
- **Nagranie z German Open** — wykorzystane tylko jako obserwacja, że wciskanie odbywa się
  zamkniętym chwytakiem. Orientacji narzędzia z nagrania **nie da się zmierzyć**, więc hipoteza
  o pozie startowej „w tablicę" pozostaje kandydatem do sprawdzenia, nie wnioskiem (6.33).

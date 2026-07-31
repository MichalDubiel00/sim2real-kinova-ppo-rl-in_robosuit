# Plan meetingu z prof. Ferreinem (tydzień 13.07.2026)

Cel meetingu: pokazać, że część symulacyjna jest **zamknięta z wynikiem 100 %**,
i uzgodnić konkrety potrzebne do transferu sim2real (dostęp do sprzętu, kamera,
stos sterowania, zakres pracy, terminy).

---

## 1. Status — co pokazać (max 5 minut)

Otwarcie (jedno zdanie): *„Die Simulationsphase ist abgeschlossen — die trainierte
Policy löst die Aufgabe in 100 von 100 Testepisoden. Heute möchte ich den
Sim2Real-Transfer konkret planen."*

### Historia w 3 krokach (pokazuje metodę, nie tylko wynik)

1. **Środowisko**: omniboard odtworzony z mesha i zdjęcia z RoboCup (krzyż,
   4 przyciski na skosach 45° + 1 frontowy), fizyczne przyciski ze sprężyną
   (próg aktywacji 12 mm), robot Kinova Gen3 + Robotiq85 na wysokości bazy Spota.
   Obserwacje **od początku projektowane pod sprzęt**: polityka widzi tylko to,
   co da się odtworzyć na robocie (propriocepcja + względna pozycja celu z szumem
   symulującym błąd AprilTag), bez uprzywilejowanych stanów symulatora.
2. **Diagnoza i naprawa rewardu** (najciekawszy wątek inżynierski): reward v1
   utknął w lokalnym optimum „wiszenia" przy przycisku (plateau od 2.35M kroków,
   ~1/5 przycisków). Rozwiązanie: potential-based shaping (v2) → pierwsze pełne
   sukcesy; potem analiza 100 epizodów wykryła wyuczone „szorowanie" ramieniem
   po panelu (96 % epizodów!) → fine-tuning z karą kolizji ×10 (v2.1).
3. **Wynik końcowy (8M kroków)**: **100/100 sukcesów** (5/5 przycisków, losowa
   kolejność, losowa pozycja boardu ±5 cm), średnio 18 s/zadanie, kroki kolizji
   z płytą −85 % (2.4 kroku/epizod ≈ sam moment wciskania). Wideo:
   `v5_finetuned_8M.mp4`. Porównanie modeli z testem istotności (Fisher,
   p ≈ 2·10⁻⁴): `training/compare_models.py`.

Liczby pod ręką (gdyby pytał):

| Metryka (100 epizodów) | v2 @ 6M | v2.1 @ 8M |
|---|---|---|
| Sukcesy 5/5 | 87 % | **100 %** |
| Średni czas sukcesu | 26 s | **18 s** |
| Kroki kolizji z płytą / epizod | 15.8 | **2.4** |

Aktualizacja (18–19.07): wykryty **trzeci reward hack** — model częściowo
wciskał przyciski nadgarstkiem od boku (na sprzęcie: ryzyko kamery
nadgarstkowej Gen3 + detekcja wciśnięć w TCP by tego nie zaliczyła). Reward
v2.2: wciśnięcie liczy się tylko z kontaktem palców z grzybkiem (lustro
detekcji nadzorcy), kara za każdy nie-palcowy kontakt z przyciskiem, wymuszony
zamknięty gripper. Fine-tuning v2.2 **zakończył się porażką pouczającą**
(0/100 sukcesów): kara odpalała się też przy poprawnych dociskach (knuckle
chwytaka muska grzybek), więc polityka nauczyła się unikać przycisków w ogóle.
Reward v2.3 zawęża karę do ogniw ramienia (kontakt chwytaka wolny), bramka
palcowa zostaje. Wynik fine-tuningu v2.3 (100 epizodów): **85/100 sukcesów**
(stary model w tym env: 59/100, Fisher p ≈ 7·10⁻⁵), czas 23 s, misuse
ramienia −75 %, kolizje −85 %; kontynuacja do 12M: plateau 85–87 %.
Analiza sparowana: porażki dwóch najlepszych modeli niemal rozłączne
(2/100 wspólnych) → brak systematycznie trudnych konfiguracji; na sprzęcie
nadzorca może dodać retry (porażki to timeouty, nie stany terminalne). Dobra historia na meeting: nagroda musi
odrzucać dokładnie te zachowania, których nie zaliczy sprzętowy detektor —
**i żadnych więcej**, bo nadmiarowa kara wygasza samo zachowanie docelowe.

---

## 2. Proponowany plan sim2real (do przedstawienia)

Pipeline (już opisany w dokumentacji, rozdz. 5):

```
kamera → AprilTag: poza 6D boardu → pozycje/osie 5 przycisków z geometrii CAD
      → mapowanie na obs polityki (target_relative_pos, press_axis, one-hot)
      → polityka PPO → delty OSC → kontroler kartezjański Kinovy (Kortex)
detekcja wciśnięcia: siła z momentów stawów + głębokość z kinematyki (≥12 mm)
      → maszyna stanów nadzorcy przełącza cel na następny przycisk
```

Proponowane etapy (każdy z osobna testowalny):

1. **Percepcja bez robota**: AprilTag na boardzie, estymacja pozy 6D, pomiar
   rzeczywistego błędu → kalibracja szumu percepcji w symulacji do zmierzonych
   wartości (ew. dotrening).
2. **Sterowanie bez polityki**: most obs/akcje ↔ Kortex API; ręcznie zadawane
   delty kartezjańskie, limity prędkości/przyspieszeń, przestrzeń robocza,
   zatrzymanie awaryjne. Weryfikacja zgodności układów współrzędnych.
3. **Polityka na robocie, tryb „powietrze"**: board wirtualny (poza z AprilTag,
   ale cel przesunięty przed płytę) — obserwacja trajektorii bez ryzyka kolizji.
4. **Pełne zadanie**: dojazd + wciskanie, detekcja z momentów stawów, sekwencja
   5 przycisków pod maszyną stanów.
5. (Opcjonalnie, jako rozdział porównawczy) percepcja YOLO+depth zamiast markera.

Szacunek czasu: ~2 tygodnie na etapy 1–2 (można zacząć od razu po uzyskaniu
dostępu do sprzętu), etapy 3–4 zależne od dostępności robota.

### Architektura nadzorcy (gdy padnie pytanie „jak konkretnie")

Zasada: **nadzorca = wszystko, co w symulacji robi env poza fizyką** — RL
odpowiada tylko za ruch, logika zadania i bezpieczeństwo są deterministyczne
i audytowalne. Jedna pętla 20 Hz (jak `control_freq` w sim):

1. raz na epizod: AprilTag → poza 6D boardu → pozycje/osie przycisków z CAD,
2. co cykl: odczyt Kinovy → **detektor wciśnięcia** (głębokość z kinematyki
   ≥ ~10–12 mm w TCP **i** siła ~4–5 N wzdłuż osi z `tool_external_wrench`
   Kortexa, debounce 3–5 cykli) → maszyna stanów przełącza cel → budowa obs
   (bit w bit jak w sim) → polityka → warstwa bezpieczeństwa → komenda
   kartezjańska.

Warstwa bezpieczeństwa poza polityką: box roboczy, limity prędkości, próg
siły → stop, E-Stop operatora. Szczegóły + tabela odpowiedniości sim↔real:
dokumentacja, rozdz. 5.1.

**Status (19.07): nadzorca ZAIMPLEMENTOWANY i przetestowany bez robota**
(moduł `supervisor/`, ~550 linii; symulacja podpięta w miejsce Kortexa przez
ten sam interfejs). Test akceptacyjny 100 epizodów: pętla nadzorcy odtwarza
wynik polityki (84 vs 85/100), a mechanizm **retry** (brak wciśnięcia 20 s →
odjazd i ponowny dojazd) podnosi sukces do **92/100** przy budżecie 75 s —
porażki polityki są przejściowe, nie terminalne. Ważny wynik negatywny:
detekcja wciśnięcia z siły+kinematyki TCP okazała się niewykonalna dla stylu
docisku obecnej polityki (pomiary w docs 5.2) → na sprzęcie detekcją
pierwszego wyboru będzie zatrzask E-Stopa (spadek siły / wizja).
Do labu zostaje TYLKO backend KortexIO (odczyt stanu + komenda twist).

**Update (21.07): próba naprawy gracji kontaktu (reward v2.4) — PORAŻKA
z pouczającą przyczyną.** Kary za obijanie się o zły przycisk i za prędkość
kontaktu obniżyły sukces ze 85% do 70% (przy dalszym treningu do 43%),
a jakość kontaktu wcale się nie poprawiła. Właściwa przyczyna (zweryfikowana
pomiarem): maszyna stanów przeskakuje cel w kroku sukcesu, a palec jeszcze
przez kilkanaście kroków dotyka wciśniętego przycisku podczas odjazdu —
**93% "złych kontaktów" to ten artefakt**, więc kary odpalały się tuż po
każdym udanym wciśnięciu i "skubały" bonus (RL: wciskanie przestaje się
opłacać). Dobra historia na meeting o reward designie: kara sprzężona
w czasie z zachowaniem docelowym wygasza to zachowanie, nawet jeśli
semantycznie miała karać co innego. Poprawiony reward v2.5 (kary tylko za
kontakt z niewciśniętym nie-celem) też odrzucony: best 83/100 (poniżej bazy
90/100), a prawdziwe obijanie okazało się mieć bazę zaledwie ~2% kontaktów
— kara nie ma na czym pracować. Po trzech porażkach (v2.2/v2.4/v2.5)
wniosek do dyskusji na meetingu: gracji kontaktu nie wymuszać rewardem,
tylko limitem prędkości TCP przy tablicy w warstwie bezpieczeństwa nadzorcy
(działa identycznie w sim i na sprzęcie, nie psuje polityki). Model do
sprzętu: v2.3 (`ppo_board_v23_12M.zip`, 90/100). Szczegóły: dokumentacja,
rozdz. 6.5-6.6.

---

## 3. Pytania do profesora (właściwy cel meetingu)

Sprzęt i dostęp:
1. **Dostęp do Kinovy Gen3** — kiedy i gdzie (lab MASCOR? robot na Spocie czy
   wolnostojący?), jak wygląda rezerwacja/nadzór, czy potrzebne szkolenie BHP?
2. **Fizyczny omniboard z RoboCup** — czy jest dostępny i można go zamontować
   na ścianie/stelażu na czas testów?
3. **Kamera** — czy jest RealSense/inna kamera zewnętrzna, czy używać kamery
   nadgarstkowej Kinovy? Czy marker AprilTag na boardzie jest akceptowalny
   (na start), z YOLO jako rozszerzeniem?

Software:
4. **Stos sterowania** — czy zespół ma istniejący stack do Kinovy (ROS/ROS2,
   kortex_driver, sterowanie kartezjańskie)? Wolę nie budować mostu od zera,
   jeśli coś już jest z RoboCup.
5. Czy są ograniczenia/konwencje labu co do trybu sterowania (twist/impedancja,
   częstotliwość komend)?

Zakres pracy i formalności:
6. Czy zakres „transfer z AprilTag + analiza jakości transferu" **wystarcza na
   pracę inżynierską**, z YOLO jako opcją? (ustalić minimalny akceptowalny wynik:
   czy sukces = działająca sekwencja na sprzęcie, czy wystarczy rzetelna analiza
   częściowego transferu?)
7. **Harmonogram formalny** — do kiedy rejestracja/oddanie pracy, kolokwium;
   jakie kamienie milowe profesor chce widzieć i w jakim rytmie raportować
   (propozycja z mojej strony: krótki mail co 2 tygodnie).

---

## 4. Ryzyka do uczciwego wymienienia (gdy zapyta „co może pójść nie tak")

- **Gap dynamiki**: OSC w symulacji ≠ kontroler kartezjański Kortex (inna
  impedancja, opóźnienia). Mitygacja: kara za zmianę akcji już w treningu,
  limity prędkości po stronie robota, etap „w powietrzu" przed kontaktem.
- **Detekcja wciśnięcia z momentów stawów** — nieprzetestowana; może wymagać
  progowania siły po kalibracji grawitacji. Plan B: weryfikacja wizyjna
  (prawdziwy E-Stop zatrzaskuje się ~10 mm niżej — widoczne na kamerze).
- **Geometria chwytaka**: polityka wciska zamkniętym chwytakiem Robotiq85 —
  sprawdzić, czy realny TCP odpowiada symulacyjnemu (offset do skalibrowania).
- Dostępność robota to główny czynnik terminowy — stąd pytania 1–2.

---

## 5. Checklist przed meetingiem

- [ ] Wideo `v5_finetuned_8M.mp4` odpalone lokalnie i gotowe do pokazania
  (nie polegać na załączniku z maila)
- [ ] `docs/img/v4_eval_curve.png` pod ręką (krzywa uczenia)
- [ ] Tabelka wyników v2 vs v2.1 (wyżej) — ew. wydruk/notatka
- [ ] Laptop: `python training/watch_board.py` działa (demo na żywo, jeśli
  będzie czas i chęć)
- [ ] Kalendarz: własna dostępność na najbliższe tygodnie (żeby od razu
  umówić dostęp do labu)

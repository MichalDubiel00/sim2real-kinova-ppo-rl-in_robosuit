"""
Detektor wcisniecia (docs 5.1): glebokosc z kinematyki + sila + debounce
=======================================================================
Board nie jest podlaczony elektrycznie — wcisniecie wykrywamy z:
  1. glebokosci: TCP przekroczyl spoczynkowy szczyt grzybka wzdluz osi
     wciskania (estymata obarczona bledem pozy boardu, stad prog < 12 mm),
  2. sily wzdluz osi (sprezyna 500 N/m x 12 mm ~ 6 N przy pelnym skoku),
  3. debounce N cykli (odpornosc na szum sily i drgania estymaty).
Zdarzenie zglaszane raz (zbocze narastajace); reset przy zmianie celu.

WNIOSEK KALIBRACYJNY (2026-07-19, 8 epizodow sladow, logs/board/
detector_traces.npz): dla OBECNEGO stylu wciskania polityki v2.3 detekcja
sila+kinematyka TCP jest NIEWYKONALNA — polityka naciera na przycisk z sila
>20 N przez dziesiatki-setki krokow przed aktywacja (brak lokalizacji w
czasie), TCP bywa 4-10 cm od osi przycisku (glebokosc absolutna: blad
+/-37 mm), a skok grzybka nie przenosi sie na ruch TCP (delta glebokosci:
MISS 23-33/36). Na sprzecie detekcje ratuje ZATRZASK prawdziwego E-Stopa
(spadek sily po zatrzasnieciu / weryfikacja wizyjna) — patrz
SimTruthDetector; docelowo: reward v2.4 wymuszajacy wykrywalny styl docisku
(bramka centrowania TCP na osi przycisku).
"""

import numpy as np


class PressDetector:

    def __init__(self, depth_thresh=0.009, force_thresh=2.5, debounce=3):
        self.depth_thresh = depth_thresh
        self.force_thresh = force_thresh
        self.debounce = debounce
        self.reset()

    def reset(self):
        self._count = 0
        self._fired = False

    def update(self, tcp_pos, btn_top_rest, press_axis, wrench_force) -> bool:
        """Jeden cykl 20 Hz; True = wykryto wcisniecie (jednorazowo)."""
        depth = float(np.dot(np.asarray(tcp_pos) - btn_top_rest, press_axis))
        force = abs(float(np.dot(wrench_force, press_axis)))
        if depth >= self.depth_thresh and force >= self.force_thresh:
            self._count += 1
        else:
            self._count = 0
        if self._count >= self.debounce and not self._fired:
            self._fired = True
            return True
        return False


class SimTruthDetector:
    """Odpowiednik zatrzasku E-Stopa / detekcji wizyjnej na czas testow
    SimIO: wcisniecie raportowane, gdy fizyka srodowiska je zaliczy
    (prawdziwy E-Stop zatrzaskuje sie ~10 mm nizej — zdarzenie obserwowalne
    przez spadek sily i na kamerze, wiec to NIE jest sygnal nieosiagalny
    na sprzecie; niedostepna jest jedynie jego rekonstrukcja z sily+TCP,
    patrz wniosek kalibracyjny wyzej)."""

    def __init__(self, sim_io):
        self.io = sim_io
        self._last = 0

    def reset(self):
        self._last = self.io.env._current_idx

    def update(self, tcp_pos, btn_top_rest, press_axis, wrench_force) -> bool:
        cur = self.io.env._current_idx
        fired = cur > self._last
        self._last = cur
        return fired

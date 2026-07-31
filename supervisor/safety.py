"""
Warstwa bezpieczenstwa — poza polityka, deterministyczna (docs 5.1)
===================================================================
Filtruje KAZDA komende przed wyslaniem do robota:
  - clip akcji do [-1,1] (limit predkosci kartezjanskiej),
  - box roboczy: skladowe pozycyjne wypychajace TCP poza box sa zerowane
    (robot moze zawsze wrocic, nigdy dalej wyjsc),
  - prog sily: przekroczenie => stop (akcja zerowa), licznik zdarzen.
Na sprzecie dochodzi E-Stop operatora (poza softwarem).
"""

import numpy as np


class SafetyLayer:

    def __init__(self,
                 box=((0.02, 0.485), (-0.55, 0.55), (0.68, 1.70)),
                 force_stop=25.0):
        """box: granice TCP (x przed murem WALL_X=0.50); force_stop [N] —
        prog sensowny na sprzecie (sprezyna ~6 N); w SYMULACJI sily
        kontaktowe sa nierealistyczne (80-400 N przy legalnym docisku),
        wiec testy SimIO wylaczaja go przez force_stop=inf."""
        self.box = np.asarray(box, dtype=np.float64)
        self.force_stop = force_stop
        self.force_stops = 0

    def reset(self):
        self.force_stops = 0

    def filter(self, action6, tcp_pos, wrench_force) -> np.ndarray:
        act = np.clip(np.asarray(action6, dtype=np.float64), -1.0, 1.0)
        if np.linalg.norm(wrench_force) > self.force_stop:
            self.force_stops += 1
            return np.zeros(6)
        for ax in range(3):
            lo, hi = self.box[ax]
            if tcp_pos[ax] <= lo and act[ax] < 0:
                act[ax] = 0.0
            if tcp_pos[ax] >= hi and act[ax] > 0:
                act[ax] = 0.0
        return act

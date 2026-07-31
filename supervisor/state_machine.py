"""
Maszyna stanow zadania + mechanizm retry
========================================
TaskStateMachine: sekwencja celow (kolejnosc = specyfikacja zadania,
na sprzecie od operatora). RetryManager: porazki polityki sa przejsciowe,
nie terminalne (analiza sparowana eval100_v23_12M: 2/100 wspolnych porazek
dwoch modeli) — brak wcisniecia przez `timeout` s => odjazd do punktu
przed celem i oddanie sterowania polityce na swiezo.
"""

import numpy as np

POLICY = "POLICY"
RETREAT = "RETREAT"


class TaskStateMachine:

    def __init__(self, n_buttons=5):
        self.n_buttons = n_buttons
        self.reset(list(range(n_buttons)))

    def reset(self, order):
        self.order = list(order)
        self.idx = 0
        self.pressed = [False] * self.n_buttons

    @property
    def done(self):
        return self.idx >= self.n_buttons

    def current_target(self):
        return None if self.done else self.order[self.idx]

    def on_press(self):
        self.pressed[self.current_target()] = True
        self.idx += 1

    def onehot(self):
        oh = np.zeros(self.n_buttons, dtype=np.float32)
        if not self.done:
            oh[self.current_target()] = 1.0
        else:  # jak env po komplecie: ostatni przycisk (nie wystapi w petli)
            oh[self.order[-1]] = 1.0
        return oh


class RetryManager:

    def __init__(self, control_freq=20.0, timeout=10.0,
                 retreat_dist=0.12, arrive_eps=0.03, retreat_max=3.0,
                 delta_scale=0.05):
        """delta_scale: ile metrow odpowiada akcji 1.0 (output_max OSC)."""
        self.freq = control_freq
        self.timeout_steps = int(timeout * control_freq)
        self.retreat_dist = retreat_dist
        self.arrive_eps = arrive_eps
        self.retreat_max_steps = int(retreat_max * control_freq)
        self.delta_scale = delta_scale
        self.reset()

    def reset(self):
        self.mode = POLICY
        self._since_progress = 0
        self._retreat_steps = 0
        self._waypoint = None
        self.retries = 0

    def on_progress(self):
        """Wcisniecie celu — licznik od nowa."""
        self._since_progress = 0

    def update(self, tcp_pos, btn_top, press_axis):
        """Jeden cykl; zwraca aktualny tryb (POLICY/RETREAT)."""
        if self.mode == POLICY:
            self._since_progress += 1
            if self._since_progress >= self.timeout_steps:
                self.mode = RETREAT
                self.retries += 1
                self._retreat_steps = 0
                self._waypoint = btn_top - press_axis * self.retreat_dist
        else:
            self._retreat_steps += 1
            arrived = np.linalg.norm(self._waypoint - tcp_pos) < self.arrive_eps
            if arrived or self._retreat_steps >= self.retreat_max_steps:
                self.mode = POLICY
                self._since_progress = 0
        return self.mode

    def retreat_action(self, tcp_pos) -> np.ndarray:
        """P-regulator kartezjanski do waypointu (rotacje bez zmian)."""
        err = self._waypoint - np.asarray(tcp_pos)
        act = np.zeros(6)
        act[:3] = np.clip(err / self.delta_scale, -1.0, 1.0) * 0.7
        return act

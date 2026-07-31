"""
Percepcja boardu: poza 6D (raz na epizod) + geometria CAD -> cele
================================================================
Na sprzecie: AprilTag daje poze srodka krzyza; pozycje/osie 5 przyciskow
wynikaja z geometrii CAD (tabela BUTTON_FACES). Szum per krok symuluje
niepewnosc estymaty TCP/tagu — te same stale co w treningu polityki.
"""

import numpy as np


class BoardPerception:

    def __init__(self, button_faces, cap_offset,
                 pos_noise=0.002, axis_noise=0.02):
        """button_faces: tabela CAD [(offset_xyz, normalna), ...] (jak w env);
        cap_offset: szczyt grzybka wzgledem srodka skosu wzdluz normalnej [m];
        pos_noise/axis_noise: stale szumu jak w treningu (ppo_board.yaml)."""
        self.button_faces = button_faces
        self.cap_offset = cap_offset
        self.pos_noise = pos_noise
        self.axis_noise = axis_noise
        self.board_pose = None
        self._axis_bias = np.zeros(3)

    def reset(self, board_pose_meas):
        """Nowy epizod: zapamietaj zmierzona poze boardu, wylosuj blad osi
        (stala niepewnosc orientacji tagu na caly epizod)."""
        self.board_pose = np.asarray(board_pose_meas, dtype=np.float64)
        self._axis_bias = np.random.normal(0.0, self.axis_noise, size=3)

    def btn_top(self, i) -> np.ndarray:
        """Estymowany szczyt grzybka przycisku i (stan spoczynkowy)."""
        offset, normal = self.button_faces[i]
        return (self.board_pose + np.asarray(offset, dtype=np.float64)
                + np.asarray(normal, dtype=np.float64) * self.cap_offset)

    def press_axis(self, i) -> np.ndarray:
        """Estymowany kierunek wciskania (do wnetrza boardu), z bledem epizodu."""
        axis = -np.asarray(self.button_faces[i][1], dtype=np.float64) \
            + self._axis_bias
        return axis / np.linalg.norm(axis)

    def perceived_rel(self, i, tcp_pos) -> np.ndarray:
        """Wektor TCP->cel jak widzi go percepcja: estymata + szum per krok.
        Bias pozy boardu siedzi juz w btn_top (blad pomiaru AprilTag)."""
        noise = np.random.normal(0.0, self.pos_noise, size=3)
        return self.btn_top(i) - np.asarray(tcp_pos) + noise

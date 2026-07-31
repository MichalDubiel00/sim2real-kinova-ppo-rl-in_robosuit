"""
Interfejs robota dla nadzorcy (docs rozdz. 5.1)
==============================================
RobotIO to JEDYNA granica miedzy nadzorca a sprzetem/symulacja:
- SimIO  (sim_io.py)  — ButtonBoardEnv w miejscu robota (testy bez sprzetu)
- KortexIO (przyszlosc) — Kinova Kortex API (twist kartezjanski,
  BaseCyclic feedback, tool_external_wrench)
Nadzorca nie widzi niczego poza tym interfejsem.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class RobotState:
    """Stan robota w jednym cyklu petli 20 Hz."""
    tcp_pos: np.ndarray      # (3,) [m], pozycja TCP w ukladzie bazy/swiata
    tcp_quat: np.ndarray     # (4,) xyzw, orientacja TCP
    wrench_force: np.ndarray  # (3,) [N], sila zewnetrzna na TCP (uklad swiata)
    proprio: dict            # nazwane obserwacje proprio (robot0_*) do obs buildera


class RobotIO(ABC):
    """Abstrakcja sprzetu: odczyt stanu + komenda kartezjanska co cykl."""

    control_freq: float = 20.0

    @abstractmethod
    def reset(self) -> RobotState:
        """Przygotowanie epizodu (sim: env.reset; hw: dojazd do pozy startowej)."""

    @abstractmethod
    def apply(self, delta6: np.ndarray) -> RobotState:
        """Wykonaj 6D delte kartezjanska [-1,1] (gripper zawsze zamkniety),
        zwroc stan po cyklu sterowania."""

    @abstractmethod
    def episode_running(self) -> bool:
        """False gdy epizod przerwany (sim: done/horizon; hw: stop operatora)."""

    @abstractmethod
    def measure_board_pose(self) -> np.ndarray:
        """(3,) pozycja srodka krzyza omniboarda — na sprzecie: detekcja
        AprilTag (raz na epizod); w sim: prawda + blad pomiaru."""

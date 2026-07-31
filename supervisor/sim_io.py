"""
SimIO — ButtonBoardEnv jako backend RobotIO
===========================================
Symulacja podpieta w miejsce Kortexa: ten sam interfejs, ktory dostanie
sprzet. Nadzorca testowany na SimIO jest gotowy na KortexIO bez zmian.

Sila zewnetrzna (odpowiednik tool_external_wrench Kinovy) liczona jest
z kontaktow MuJoCo na geomach palcow (mj_contactForce); gdy API niedostepne,
fallback: 6 N przy kontakcie palca z grzybkiem (model sprezyny 500 N/m x 12 mm).
"""

import numpy as np

from .interfaces import RobotIO, RobotState


class SimIO(RobotIO):

    def __init__(self, env, board_meas_err=0.005):
        """env: ButtonBoardEnv; board_meas_err: symulowany blad AprilTag [m]."""
        self.env = env
        self.board_meas_err = board_meas_err
        self.control_freq = env.control_freq
        self._pf = env.robots[0].robot_model.naming_prefix
        self._done = False
        self._obs = None

    # -- RobotIO ------------------------------------------------------

    def reset(self) -> RobotState:
        self._obs = self.env.reset()
        self._done = False
        # hard_reset przebudowuje model MuJoCo — geom IDs trzeba odswiezyc
        model = self.env.sim.model
        self._robot_geoms = {
            gid for gid in range(model.ngeom)
            if (model.geom_id2name(gid) or "").startswith(("robot", "gripper"))
        }
        return self._state()

    def apply(self, delta6) -> RobotState:
        action = np.append(np.asarray(delta6, dtype=np.float64), 1.0)
        self._obs, _, self._done, _ = self.env.step(action)
        return self._state()

    def episode_running(self) -> bool:
        return not self._done

    def measure_board_pose(self) -> np.ndarray:
        """Srodek krzyza jak widzialby go AprilTag: prawda + bias pomiaru."""
        true_pose = np.array([self.env.WALL_X, self.env._board_y,
                              self.env._board_z])
        bias = np.random.uniform(-self.board_meas_err, self.board_meas_err, 3)
        return true_pose + bias

    # -- pomocnicze (tylko sim) ---------------------------------------

    def task_order(self):
        """Kolejnosc przyciskow = specyfikacja zadania (na sprzecie: operator)."""
        return list(self.env._order)

    def calibrate_cap_offset(self) -> float:
        """Stala CAD: odleglosc szczytu grzybka od srodka skosu wzdluz
        normalnej (odczyt jednorazowy, na sprzecie z modelu CAD)."""
        env = self.env
        top = env.sim.data.site_xpos[env._btn_top_ids[0]]
        body = np.array(env._btn_body_pos(0))
        normal = np.asarray(env.BUTTON_FACES[0][1], dtype=np.float64)
        return float(np.dot(top - body, normal))

    def _state(self) -> RobotState:
        obs = self._obs
        return RobotState(
            tcp_pos=np.asarray(obs[f"{self._pf}eef_pos"]).flatten(),
            tcp_quat=np.asarray(obs[f"{self._pf}eef_quat"]).flatten(),
            wrench_force=self._finger_force(),
            proprio=obs,
        )

    def _finger_force(self) -> np.ndarray:
        """Suma ZEWNETRZNYCH sil kontaktowych na palcach (uklad swiata).
        Kontakty palec x robot (docisk zamknietych palcow do siebie,
        knuckle itd.) sa wewnetrzne — tool_external_wrench tez ich nie
        widzi — wiec je pomijamy."""
        sim = self.env.sim
        fingers = self.env._finger_geom_ids
        total = np.zeros(3)
        raw_m = getattr(sim.model, "_model", None)
        raw_d = getattr(sim.data, "_data", None)
        for i, c in enumerate(sim.data.contact[: sim.data.ncon]):
            g1f, g2f = c.geom1 in fingers, c.geom2 in fingers
            if not (g1f or g2f):
                continue
            other = c.geom2 if g1f else c.geom1
            if other in self._robot_geoms:
                continue  # kontakt wewnetrzny chwytaka/robota
            if raw_m is not None and raw_d is not None:
                import mujoco
                f6 = np.zeros(6)
                mujoco.mj_contactForce(raw_m, raw_d, i, f6)
                # frame: wiersze = osie ukladu kontaktu (x = normalna)
                total += c.frame.reshape(3, 3).T @ f6[:3]
            else:  # fallback: model sprezyny przy kontakcie z grzybkiem
                if other in set(self.env._cap_geom_ids):
                    total += 6.0 / np.sqrt(3.0)
        return total

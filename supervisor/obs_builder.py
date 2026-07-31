"""
Budowa wektora obserwacji — bit w bit jak w treningu
====================================================
Polityka widzi flatten(obs_dict.values()) w kolejnosci kluczy robosuite.
Na sprzecie proprio (robot0_*) liczy KortexIO z feedbacku Kinovy, a czesc
"object" buduje nadzorca z percepcji + maszyny stanow. Szablon kolejnosci
kluczy pobierany jest raz z przykladowego obs env (i zapisywalny do pliku
— na sprzecie musi byc identyczny z treningowym).
"""

import numpy as np
import robosuite.utils.transform_utils as T

OBJECT_KEYS = [
    "target_relative_pos",
    "target_distance",
    "target_onehot",
    "target_press_axis",
    "eef_align_with_target",
]


class ObsBuilder:

    def __init__(self, template_keys, naming_prefix="robot0_"):
        """template_keys: kolejnosc kluczy obs_dict z treningu (lista)."""
        self.keys = list(template_keys)
        self.pf = naming_prefix

    def object_obs(self, perception, sm, tcp_pos, tcp_quat) -> dict:
        """Czesc 'object' liczona przez nadzorce (nie env!)."""
        tgt = sm.current_target()
        if tgt is None:  # po komplecie (nie wystapi w petli — dla kompletnosci)
            tgt = sm.order[-1]
        rel = perception.perceived_rel(tgt, tcp_pos)
        axis = perception.press_axis(tgt)
        eef_z = T.quat2mat(np.asarray(tcp_quat))[:, 2]
        cos_a = float(np.clip(np.dot(eef_z, axis), -1.0, 1.0))
        return {
            "target_relative_pos": rel,
            "target_distance": np.array([np.linalg.norm(rel)]),
            "target_onehot": sm.onehot(),
            "target_press_axis": axis.astype(np.float32),
            "eef_align_with_target": np.array([np.arccos(abs(cos_a))]),
        }

    def build(self, proprio, obj) -> np.ndarray:
        """Sklej pelny wektor w kolejnosci szablonu; klucze object i agregat
        'object-state' pochodza od nadzorcy, reszta z proprio robota."""
        parts = []
        for k in self.keys:
            if k in obj:
                v = obj[k]
            elif k == "object-state":
                v = np.concatenate([np.asarray(obj[n]).flatten()
                                    for n in OBJECT_KEYS])
            else:
                v = proprio[k]
            parts.append(np.asarray(v).flatten())
        return np.concatenate(parts, dtype=np.float32)

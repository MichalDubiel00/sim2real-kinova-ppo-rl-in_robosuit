import os
from robosuite.models.objects import MujocoXMLObject

ASSET_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "../my_assets/objects"))

# Liczba slotow na boardzie
N_SLOTS = 5


class OmniBoardObject(MujocoXMLObject):
    """Krzyzowy panel Omniboard (Omniboard.stl, 320x320x80 mm).

    Body origin = srodek krzyza na frontowej (duzej) scianie;
    glebokosc plyty rozciaga sie w lokalnym +Z.
    Przyciski montowane sa osobno przez srodowisko (nie sa czescia meshu).
    """

    def __init__(self, name="omniboard"):
        super().__init__(
            fname=os.path.join(ASSET_DIR, "xml/omniboard.xml"),
            name=name,
            joints=None,          # statyczny obiekt
            obj_type="all",
            # False: XML ma wlasne geomy wizualne; duplikaty kolizji
            # renderowaly sie jako szare bryly zaslaniajace mesh
            duplicate_collision_geoms=False,
        )


class linearboardObject(MujocoXMLObject):
    """Board z 5 cylindrami (linear_board.stl + 50mm_pipe v1.stl).

    Sites dla kazdego slotu (i = 0..4):
      - linearboard_slot_{i}_opening   : otwar cylindra (cel dla kamery)
      - linearboard_slot_{i}_axis_top  : punkt na osi cylindra (do liczenia kata)
    """

    def __init__(self, name="linearboard"):
        super().__init__(
            fname=os.path.join(ASSET_DIR, "xml/linearboard.xml"),
            name=name,
            joints=None,          # statyczny obiekt
            obj_type="all",
            duplicate_collision_geoms=True,
        )

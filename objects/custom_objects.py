import os
from robosuite.models.objects import MujocoXMLObject

ASSET_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "../my_assets/objects"))

class OmniBoardObject(MujocoXMLObject):
    """Cross-shaped Omniboard panel (Omniboard.stl, 320x320x80 mm). Origin = cross
    centre on the front face, depth along local +Z. Buttons are added by the env."""

    def __init__(self, name="omniboard"):
        super().__init__(
            fname=os.path.join(ASSET_DIR, "xml/omniboard.xml"),
            name=name,
            joints=None,          # static object
            obj_type="all",
            # False: the XML has its own visual geoms; collision duplicates
            # rendered as grey solids covering the mesh
            duplicate_collision_geoms=False,
        )


import os
from robosuite.models.objects import MujocoXMLObject

ASSET_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "../my_assets/objects"))


class CylinderObject(MujocoXMLObject):
    """
    50mm inspection cylinder. The EEF wrist camera should look into the top opening.

    Key sites in the XML:
      - target_bottom : center of the cylinder opening (z-axis points up = into the tube)
    """

    def __init__(self, name):
        super().__init__(
            fname=os.path.join(ASSET_DIR, "xml/cylinder.xml"),
            name=name,
            joints=[{"type": "free", "damping": "0.0005"}],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

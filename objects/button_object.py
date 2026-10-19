import os
from robosuite.models.objects import MujocoXMLObject

ASSET_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "../my_assets/objects"))


class ButtonObject(MujocoXMLObject):
    """
    Emergency Stop button with a spring-loaded prismatic joint.

    Key sites:
      - button_top          : target for EEF approach (top of the cap)
      - button_contact_site : force sensor site (read via MjSim.data.sensordata)
    Key joint:
      - button_slide        : prismatic joint, range [0, 0.015] m
                              activation threshold: 0.012 m
    """

    ACTIVATION_THRESHOLD = 0.012  # metres — button considered pressed

    def __init__(self, name):
        super().__init__(
            fname=os.path.join(ASSET_DIR, "xml/button.xml"),
            name=name,
            joints=None,  # static object; spring joint is inside the body
            obj_type="all",
            # False: the XML has its own visual geoms; collision duplicates rendered
            # as grey cylinders covering them
            duplicate_collision_geoms=False,
        )

import numpy as np
from robosuite.environments.manipulation.pick_place import PickPlaceCan

class Door(PickPlaceCan):
    """
    Door novelty: an locked door is placed in between the pick and drop areas, the agent has to pull the handle.
    """

    def __init__(self, **kwargs):
        super().__init__(door_pos=(-0.20, 0.35, 0.1, -3*np.pi/2), plate_pos=(-0.4, -0.225, 0.1), door_locked=True, novelty="Door", **kwargs)


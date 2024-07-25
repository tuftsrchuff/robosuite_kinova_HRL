import numpy as np
from robosuite.personal_scripts.hanoi_env import Hanoi

class DoorNovelty(Hanoi):
    """
    Door novelty: an locked door is placed in between the pick and drop areas, the agent has to pull the handle.
    """

    def __init__(self, **kwargs):
        super().__init__(door_pos=(-0.20, 0.35, 0.1, -3*np.pi/2), door_locked=True, **kwargs)


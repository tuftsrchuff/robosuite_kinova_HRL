import numpy as np
from robosuite.environments.manipulation.hanoi import Hanoi

class DoorNovelty(Hanoi):
    """
    Door novelty: an locked door is placed in between the pick and drop areas, the agent has to pull the handle.
    """

    def __init__(self, **kwargs):
        super().__init__(door_pos=(0.12, -0.075, 0.8, -3*np.pi/2), door_locked=True, **kwargs)



import numpy as np
from robosuite.environments.manipulation.pick_place import PickPlaceCan

class Lightoff(PickPlaceCan):
    """
    Lightoff novelty: the light is turned off, the agent can't see the object/areas and return 0 as a distance to them
    """

    def __init__(self, **kwargs):
        super().__init__(light_on=False, novelty="Lightoff", **kwargs)



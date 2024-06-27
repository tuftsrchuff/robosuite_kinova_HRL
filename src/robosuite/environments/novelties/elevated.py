from robosuite.environments.manipulation.pick_place import PickPlaceCan

class Elevated(PickPlaceCan):
    """
    Elevated novelty: the drop area is elevated above the pick area.
    """

    def __init__(self, **kwargs):
        super().__init__(bin2_pos=(0.1, 0.28, 0.95), novelty="Elevated", **kwargs)


from robosuite.environments.manipulation.pick_place import PickPlaceCan

class Hole(PickPlaceCan):
    """
    Hole novelty: a plate with a hole in it is placed above the drop area.
    """

    def __init__(self, **kwargs):
        super().__init__(plate_pos=(-0.137, 0.525, 0.12), novelty="Plate", **kwargs)


from robosuite.environments.manipulation.pick_place import PickPlaceCan

class Obstacle(PickPlaceCan):
    """
    Obstacle novelty: a cylinder object that is placed in between the pick and drop areas.
    """

    def __init__(self, **kwargs):
        super().__init__(cylinder_pos=(-0.07, 0.27, 0.1), novelty="Cylinder", **kwargs)
        #cylinder_pos=(0.02, 0.27, 0.1)

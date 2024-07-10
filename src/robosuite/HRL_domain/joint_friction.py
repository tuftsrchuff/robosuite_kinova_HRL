import numpy as np
import robosuite as suite
from robosuite.utils.mjmod import DynamicsModder

class JointNovelty:
    def __init__(self, env, joint_friction):
        """
        Novelty for joint that modifies friction

        :param env: the environment the novelty extends/changes.
        """
        self.env = env
        self.joint_friction = joint_friction
        self.modder = DynamicsModder(sim=env.sim, random_state=np.random.RandomState(5))
        #TODO: Update this modder to properly update friction on all joints
        self.modder.mod(geom_name, "friction", [2.0, 0.2, 0.04])  
        self.modder.update()   

    def step(self, action, obs):
        """
        Perform a step in the novelty, called after a step in the environment was performed.

        :param obs: the observation that would be returned in that step, may be modified
        :param action: the action the agent took.
        """
        pass

    def reset(self, obs):
        """
        Reset the novelty, called after the environment was reset.

        :param obs: the observation that would be returned in that step, may be modified
        """
        pass

    def remove(self):
        """
        Remove the novelty from the environment, called in the close method of the environment,
        should be called manually when removing a novelty.
        """
        self.modder.restore_defaults()
import gymnasium as gym
import robosuite as suite
import numpy as np
from stable_baselines3 import SAC
from robosuite import load_controller_config

controller_config = load_controller_config(default_controller='OSC_POSITION')

class PickWrapper(gym.Wrapper):
    def __init__(self, env, dense_reward=True):
        # Run super method
        super().__init__(env=env)
        # Define needed variables
        self.obj_body = self.env.sim.model.body_name2id('Can_main')
        self.gripper_body = self.env.sim.model.body_name2id('gripper0_eef')
        self.env.obj_to_use = 'can'
        self.dense_reward = dense_reward
        self.count_step = 0

    def render(self, mode='human'):
        self.env.viewer.render()

    def reset(self, seed=None):
        # Reset the environment for the pick task
        obs, info = self.env.reset()
        self.env.obj_to_use = 'can'
        self.goal = self._sample_pick_goal()
        self.env.sim.forward()

        # Print out the body names to check the name of the gripper body
        print("Reseting environment...")

        # Return the initial observation
        return obs, info

    def _sample_pick_goal(self):
        # Get the current position of the 'Can' object
        object_pos = self.env.sim.data.body_xpos[self.obj_body]

        # Set the target goal to be directly above the current position of the 'Can' object
        goal = np.array([object_pos[0], object_pos[1], object_pos[2] + 0.2])  # target goal is 10cm above the current position

        return goal

    def step(self, action):
        # Perform the pick step
        obs, reward, terminated, truncated, info = self.env.step(action)
        truncated = truncated or self.env.done

        """if self.count_step == 0:
            offset = np.asarray([-0.1])
        else:
            offset = np.asarray([0.1])
        body_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_body])
        body_pos[2] = body_pos[2] + 0.05
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
        if self.count_step <=100:
            action = np.concatenate([body_pos-gripper_pos, offset])
        else:
            action = np.concatenate([body_pos, offset])
        print(action)
        for i in range(100):
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            self.env.sim.forward()
        self.count_step += 1
        """

        # Compute the reward based on the distance between the object and the target goal
        reward, terminated = self.compute_reward()
        
        return obs, reward, terminated, truncated, info

    def compute_reward(self):
        # Compute the reward based on the distance between the object and the target goal
        object_pos = self.env.sim.data.body_xpos[self.obj_body]
        target_pos = self.goal
        dist_to_target = np.linalg.norm(object_pos - target_pos)

        if self.dense_reward:
            # Compute a dense reward based on the distance to the target goal
            reward = -dist_to_target
            if dist_to_target < 0.05:
                # If the object is close to the target goal, give a bonus reward
                reward += 1.0
        else:
            # Compute a binary reward based on whether the object has been placed at the target goal
            grip_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            dist_to_object = np.linalg.norm(object_pos - grip_pos)
            if dist_to_object < 0.05 and dist_to_target < 0.05:
                # If the object has been picked up and placed at the target goal, give a reward of 0.0
                reward = 0.0
            else:
                # Otherwise, give a reward of -1.0
                reward = -1.0

        # Check if the object has been picked up and placed at the target goal
        grip_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
        dist_to_object = np.linalg.norm(object_pos - grip_pos)
        if dist_to_object < 0.05 and dist_to_target < 0.05:
            # If the object has been picked up and placed at the target goal, set done=True
            done = True
        else:
            done = False
        
        #print(object_pos, target_pos, dist_to_target, reward, done)

        return reward, done


class ReachWrapper(gym.Env):
    def __init__(self, env, pick_policy_path, dense_reward=True):
        # Run super method
        super().__init__(env=env)
        # Define needed variables
        self.obj_body = self.sim.model.body_name2id('Can_main')
        self.gripper_body = self.sim.model.body_name2id('gripper0_eef')
        self.obj_to_use = 'can'
        self.pick_policy = SAC.load(pick_policy_path)
        self.dense_reward = dense_reward

    def reset(self):
        # Reset the environment for the reach task
        obs, info = self.env.reset()
        done = False
        while not done:
            action, _states = self.pick_policy.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
        
        self._sample_reach_goal()
        self.sim.forward()

        # Return the initial observation
        return obs, info

    def _sample_reach_goal(self):
        # Get the current target position from the PickPlace instance
        target_pos = self.sim.data.body_xpos[self.sim.model.body_name2id('VisualCan_main')]

        # Add 0.1 to the z-axis of the target position
        goal_pos = np.array([target_pos[0], target_pos[1], target_pos[2] + 0.2])

        # Set the new goal position
        self.goal = goal_pos

    def step(self, action):
        # Perform the reach step
        obs, reward, terminated, truncated, info = self.env.step(action)
        truncated = truncated or self.env.done

        # Compute the reward based on the distance between the object and the target goal
        reward, terminated = self.compute_reward()
        
        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        # Compute the reward based on the distance between the gripper and the target goal
        gripper_pos = self.sim.data.body_xpos[self.obj_body]
        target_pos = self.goal
        dist_to_target = np.linalg.norm(gripper_pos - target_pos)
        done = False
        
        if self.dense_reward:
            reward = -dist_to_target
        else:
            if dist_to_target < 0.05:
                reward = 0.0
            else:
                reward = -1.0

        # Check if the object has been placed at the target goal
        object_pos = self.sim.data.body_xpos[self.sim.model.body_name2id('Can_main')]
        dist_to_object = np.linalg.norm(gripper_pos - object_pos)
        if dist_to_object < 0.05 and dist_to_target < 0.05:
            # If the object has been placed at the target goal, set done=True and give a reward of 1.0
            done = True
            if self.dense_reward:
                reward += 10
        else:
            # Check if the gripper is still holding the object
            gripper_pos = self.sim.data.body_xpos[self.sim.model.body_name2id('robot0:gripper_link')]
            object_pos = self.sim.data.body_xpos[self.sim.model.body_name2id(self.obj_to_use)]
            dist_to_object = np.linalg.norm(gripper_pos - object_pos)
            if dist_to_object > 0.1:
                # If the gripper is not holding the object, set done=True and give a negative reward
                done = True
                reward = -1.0
            else:
                done = False

        return reward, done

class PlaceWrapper(gym.Env):
    def __init__(self, env, pick_policy_path, place_policy_path, dense_reward=True):
        # Run super method
        super().__init__(env=env)
        # Define needed variables
        self.obj_body = self.sim.model.body_name2id('Can_main')
        self.gripper_body = self.sim.model.body_name2id('gripper0_eef')
        self.obj_to_use = 'can'
        self.pick_policy = SAC.load(pick_policy_path)
        self.place_policy = SAC.load(place_policy_path)
        self.dense_reward = dense_reward

    def reset(self):
        # Reset the environment for the reach task
        obs, info = self.env.reset()
        done = False
        while not done:
            action, _states = self.pick_policy.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
        
        done = False
        while not done:
            action, _states = self.place_policy.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
        
        self._sample_place_goal()
        self.sim.forward()

        # Return the initial observation
        return obs, info

    def _sample_place_goal(self):
        # Get the current target position from the PickPlace instance
        self.goal = self.sim.data.body_xpos[self.sim.model.body_name2id('VisualCan_main')]

    def step(self, action):
        # Perform the place step
        obs, reward, terminated, truncated, info = self.env.step(action)
        truncated = truncated or self.env.done

        # Compute the reward based on the distance between the object and the target goal
        reward, terminated = self.compute_reward()
        
        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        # Compute the reward based on the distance between the object and the target goal
        object_pos = self.sim.data.body_xpos[self.obj_body]
        target_pos = self.goal
        dist_to_target = np.linalg.norm(object_pos - target_pos)

        if self.dense_reward:
            # Compute a dense reward based on the distance to the target goal
            reward = -dist_to_target
            if dist_to_target < 0.05:
                # If the object is close to the target goal, give a bonus reward
                reward += 1.0
        else:
            # Compute a binary reward based on whether the object has been placed at the target goal
            if dist_to_target < 0.05:
                # If the object has been placed at the target goal, give a reward of 0.0
                reward = 0.0
            else:
                # Otherwise, give a reward of -1.0
                reward = -1.0

        # Check if the object has been placed at the target goal
        if dist_to_target < 0.05:
            # If the object has been placed at the target goal, set done=True
            done = True
        else:
            done = False

        return reward, done
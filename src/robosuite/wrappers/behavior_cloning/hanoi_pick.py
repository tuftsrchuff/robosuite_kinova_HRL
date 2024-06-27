import copy
import gymnasium as gym
import robosuite as suite
import numpy as np
from detector import Robosuite_Hanoi_Detector

controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

class PickWrapper(gym.Wrapper):
    def __init__(self, env, render_init=False, nulified_action_indexes=[], horizon=500):
        # Run super method
        super().__init__(env=env)
        self.env = env
        self.use_gripper = True
        self.render_init = render_init
        self.detector = Robosuite_Hanoi_Detector(self)
        self.nulified_action_indexes = nulified_action_indexes
        self.horizon = horizon
        self.step_count = 1

        # Define needed variables
        self.cube1_body = self.env.sim.model.body_name2id('cube1_main')
        self.cube2_body = self.env.sim.model.body_name2id('cube2_main')
        self.cube3_body = self.env.sim.model.body_name2id('cube3_main')
        self.peg1_body = self.env.sim.model.body_name2id('peg1_main')
        self.peg2_body = self.env.sim.model.body_name2id('peg2_main')
        self.peg3_body = self.env.sim.model.body_name2id('peg3_main')

        # Set reset state info:
        #self.reset_state = {'on(cube1,peg1)': True, 'on(cube2,peg3)': True, 'on(cube3,peg2)': True}
        self.reset_state = self.sample_reset_state()
        self.task = self.sample_task()
        self.env.reset_state = self.reset_state
        self.obj_mapping = {'cube1': self.cube1_body, 'cube2': self.cube2_body, 'cube3': self.cube3_body, 'peg1': self.peg1_body, 'peg2': self.peg2_body, 'peg3': self.peg3_body}
        self.goal_mapping = {'cube1': 0, 'cube2': 1, 'cube3': 2, 'peg1': 3, 'peg2': 4, 'peg3': 5}
        self.area_pos = {'peg1': self.env.pegs_xy_center[0], 'peg2': self.env.pegs_xy_center[1], 'peg3': self.env.pegs_xy_center[2]}

        # set up observation space
        self.obs_dim = self.env.obs_dim + 3 # 1 extra dimensions for the object goal

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)
        # Reduce the action space by the length of the nulified indexes (take the range low high of the action space from the env.action_space)
        self.action_space = gym.spaces.Box(low=self.env.action_space.low[:-len(nulified_action_indexes)], high=self.env.action_space.high[:-len(nulified_action_indexes)], dtype=np.float64)
        #print("Action space: ", self.action_space)
        #print("Action sample: ", self.action_space.sample())

    def search_free_space(self, cube, locations, reset_state):
        drop_off = np.random.choice(locations)
        reset_state.update({f"on({cube},{drop_off})":True})
        locations.remove(drop_off)
        locations.append(cube)
        return reset_state, locations

    def sample_reset_state(self):
        reset_state = {}
        locations = ["peg1", "peg2", "peg3"]
        cubes = ["cube3", "cube2", "cube1"]
        for cube in cubes:
            reset_state, locations = self.search_free_space(cube, locations=locations, reset_state=reset_state)
        return reset_state

    def search_valid_picks_drops(self):
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        valid_picks = []
        cubes = [3, 2, 1]
        pegs = [4, 5, 6]
        for cube in cubes:
            if state[f"clear(cube{cube})"]:
                valid_picks.append(cube)
        valid_drops = copy.copy(valid_picks)
        for peg in pegs:
            if state[f"clear(peg{peg-3})"]:
                valid_drops.append(peg)
        return valid_picks, valid_drops
    
    def sample_task(self):
        # Sample a random task
        valid_task = False
        valid_picks, valid_drops = self.search_valid_picks_drops()

        while not valid_task:
            # Sample a random task
            cube_to_pick = np.random.choice(valid_picks)
            valid_drops_copy = copy.copy(valid_drops)
            valid_drops_copy.remove(cube_to_pick)
            place_to_drop = np.random.choice(valid_drops_copy)
            if cube_to_pick >= place_to_drop:
                continue
            if place_to_drop < 4:
                place_to_drop = 'cube{}'.format(place_to_drop)
            else:
                place_to_drop = 'peg{}'.format(place_to_drop - 3)
            cube_to_pick = 'cube{}'.format(cube_to_pick)
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            if state['on({},{})'.format(cube_to_pick, place_to_drop)]:
                continue
            if state['clear({})'.format(cube_to_pick)] and state['clear({})'.format(place_to_drop)]:
                valid_task = True
        # Set the task
        self.obj_to_pick = cube_to_pick
        self.place_to_drop = place_to_drop
        #print("Task: Pick {} and drop it on {}".format(self.obj_to_pick, self.place_to_drop))
        return f"on({cube_to_pick},{place_to_drop})"

    def pick_reset(self):
        """
        Resets the environment to a state where the gripper is above the object to be picked
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state

        self.reset_step_count = 0
        #print("Moving up...")
        for _ in range(5):
            obs,_,_,_,_ = self.env.step([0,0,1,0])
            self.env.render() if self.render_init else None

        #print("Moving gripper over object...")
        while not state['over(gripper,{})'.format(self.obj_to_pick)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            action = 5*np.concatenate([dist_xy_plan, [0, 0]])
            obs,_,_,_,_ = self.env.step(action)
            self.env.render() if self.render_init else None
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 500:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        return True, obs

    def valid_state(self):
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        state = {k: state[k] for k in state if 'on' in k}
        # Filter only the values that are True
        state = {key: value for key, value in state.items() if value}
        # if state has not 3 keys, return None
        if len(state) != 3:
            return False
        # Check if cubes have fallen from other subes, i.e., check if two or more cubes are on the same peg
        pegs = []
        for relation, value in state.items():
            _, peg = relation.split('(')[1].split(',')
            pegs.append(peg)
        if len(pegs) != len(set(pegs)):
            #print("Two or more cubes are on the same peg")
            return False
        #print(state)
        return True

    def reset(self, seed=None):
        self.step_count = 1
        reset = False
        while not reset:
            trials = 0
            # Reset the environment for the pick task
            self.reset_state = self.sample_reset_state()
            self.task = self.sample_task()
            self.env.reset_state = self.reset_state
            success = False
            while not success:
                valid_state = False
                while not valid_state:
                    #print("Trying to reset the environment...")
                    try:
                        obs, info = self.env.reset()
                    except:
                        obs = self.env.reset()
                        info = {}
                    valid_state = self.valid_state()
                    trials += 1
                    if trials > 3:
                        break   
                success, obs = self.pick_reset()
                reset = success
                if trials > 3:
                    break   

        self.sim.forward()
        # replace the goal object id with its array of x, y, z location
        obs = np.concatenate((obs, self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]][:3]))
        return obs, info
    
    def step(self, action):
        # if self.nulified_action_indexes is not empty, fill the action with zeros at the indexes
        if self.nulified_action_indexes:
            for index in self.nulified_action_indexes:
                action = np.insert(action, index, 0)
        #print("Action: ", action)
        truncated = False
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except:
            obs, reward, terminated, info = self.env.step(action)
        state = self.detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=False)
        # test if self.obj_to_pick (and only self.obj_to_pick) is grasped
        state = {k: state[k] for k in state.keys() if 'grasped' in k and state[k]}
        try:
            success = len(state.keys()) == 1 and state[f'grasped({self.obj_to_pick})']
        except KeyError:
            success = False
        if success:
            print(state)
        info['is_sucess'] = success
        truncated = truncated or self.env.done
        terminated = terminated or success
        obs = np.concatenate((obs, self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]][:3]))
        reward = 1 if success else 0
        self.step_count += 1
        if self.step_count > self.horizon:
            terminated = True
        return obs, reward, terminated, truncated, info
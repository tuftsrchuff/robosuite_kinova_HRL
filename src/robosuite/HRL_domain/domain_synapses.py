import robosuite as suite
from robosuite.HRL_domain.joint_friction import JointNovelty
from robosuite.HRL_domain.detector import Detector
import numpy as np
from robosuite import load_controller_config

controller_config = load_controller_config(default_controller='OSC_POSITION')


executors = {
     'pick': './models/Pick/best_model.zip',
     'drop': './models/Drop/best_model.zip',
     'reach_drop': './models/ReachDrop/best_model.zip',
     'reach_pick': './models/ReachPick/best_model.zip'
}

obj_mapping = {}
area_pos = {}

effect_mapping = {
     'reach_pick': [True],
     'pick': [True],
     'reach_drop': [True, True],
     'drop': [True, False]
}

def effects(operator, symgoal):
     if operator == 'reach_pick':
          return [f'over(gripper,{symgoal})']
     elif operator == 'pick':
          return [f'grasped({symgoal})']
     elif operator == 'reach_drop':
        #   state[f"over(gripper,{self.place_to_drop})"] and state[f"grasped({self.obj_to_pick})"]
          return [f'over(gripper,{symgoal[1]})',f'grasped({symgoal[0]})']
     else:
        #   state[f"on({self.obj_to_pick},{self.place_to_drop})"] and not state[f"grasped({self.obj_to_pick})"]
          return [f'on({symgoal[0]},{symgoal[1]})',f'grasped({symgoal[0]})']
     

# obj_body_mapping = {

# 'o1': cube1_body,

# 'o2': cube2_body,

# 'o6': cube3_body,

# 'o3': peg1_body,

# 'o4': peg2_body,

# 'o5': peg3_body

# }

# area_pos = {'peg1': env.pegs_xy_center[0], 'peg2': env.pegs_xy_center[1], 'peg3': env.pegs_xy_center[2]}

def populateExecutorInfo(env):
     #Domain synapses here on down
        cube1_body = env.sim.model.body_name2id('cube1_main')
        cube2_body = env.sim.model.body_name2id('cube2_main')
        cube3_body = env.sim.model.body_name2id('cube3_main')
        peg1_body = env.sim.model.body_name2id('peg1_main')
        peg2_body = env.sim.model.body_name2id('peg2_main')
        peg3_body = env.sim.model.body_name2id('peg3_main')

        global obj_mapping
        global area_pos

        obj_mapping = {'cube1': cube1_body, 
                            'cube2': cube2_body, 
                            'cube3': cube3_body, 
                            'peg1': peg1_body, 
                            'peg2': peg2_body, 
                            'peg3': peg3_body}
        

        area_pos = {'peg1': env.pegs_xy_center[0], 'peg2': env.pegs_xy_center[1], 'peg3': env.pegs_xy_center[2]}


# def createObjBodyMapping(env):
#     cube1_body = env.sim.model.body_name2id('cube1_main')

#     cube2_body = env.sim.model.body_name2id('cube2_main')

#     cube3_body = env.sim.model.body_name2id('cube3_main')

#     peg1_body = env.sim.model.body_name2id('peg1_main')

#     peg2_body = env.sim.model.body_name2id('peg2_main')

#     peg3_body = env.sim.model.body_name2id('peg3_main')

#     obj_body_mapping = {

#     'o1': cube1_body,

#     'o2': cube2_body,

#     'o6': cube3_body,

#     'o3': peg1_body,

#     'o4': peg2_body,

#     'o5': peg3_body

#     }

#     return
     

# used to select source policy to transfer from whenever the agent needs to train on a novelty pattern
# novelty_patterns = {'follow_lane':[],'turn_left':[],'turn_right':[],'break':[],'park':[],'change_lane_left':[],'change_lane_right':[]}

# set up applicator mapping from operators to executors
# applicator = {'goForward':['follow_lane'],
# 		'turnLeft':['turn_left'],
# 		'turnRight':['turn_right'],
# 		'ChangeLaneLeft':['change_lane_left'],
# 		'ChangeLaneRight':['change_lane_right'],
# 		'Break':['break'],
# 		'PickUp':['pick_up'],
# 		'Drop':['drop'],
# 		'Park':['park']
# 		}


novelties_info = {
    "joint_friction": {"wrapper":JointNovelty, "params": None, "type":"global"}
}




def create_env(env_id, render=False, seed=None):
    '''
    This method is responsible for generating a new environment for the experiment. It takes an environment ID, a list of novelties, 
    a reset location and direction, a render flag, and a training flag as parameters. 

    If the environment does not exist or the novelty list has changed, it closes the existing environment (if any) and creates a new one. 
    It sets the render, reset location and direction, and environment ID attributes of the instance. 
    It determines the locality of the environment based on the novelties. If there are multiple novelties or a global novelty, the locality 
    is set to "global". If there is a single local novelty, the locality is set to "local".
    It then injects the novelties into the environment by creating a new instance of the novelty wrapper for each novelty and adding it to 
    the environment. 
    Finally, it seeds the environment and sets the environment and novelty list attributes of the instance.
    '''
    print(f"Creating env ID: {env_id}, render {render}, seed {seed}")


    # create environment instance
    env = suite.make(
        env_name="Hanoi",
        robots="Kinova3", 
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        # render_camera="agentview",
        controller_configs=controller_config,
        random_reset = True
    )

    return env

def addGoal(obs, symgoal, env, operator):
    if operator in ["reach_pick", "pick"]:
        obs = np.concatenate((obs, env.sim.data.body_xpos[obj_mapping[symgoal]][:3]))
    else:
        obs = np.concatenate((obs, env.sim.data.body_xpos[obj_mapping[symgoal[1]]][:3]))
    return obs

#Symbolic termination for beta
def termination_indicator(operator):

    if operator == 'pick':

        def Beta(env, symgoal):

            detector = Detector(env)

            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

            condition = state[f"grasped({symgoal})"]

            return condition

    elif operator == 'drop':

        def Beta(env, symgoal):

            detector = Detector(env)

            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

            condition = state[f"on({symgoal[0]},{symgoal[1]})"] and not state[f'grasped({symgoal[0]})']

            return condition

    elif operator == 'reach_pick':

        def Beta(env, symgoal):

            detector = Detector(env)

            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

            condition = state[f"over(gripper,{symgoal})"]

            return condition

    elif operator == 'reach_drop':

        def Beta(env, symgoal):

            detector = Detector(env)

            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

            condition = state[f"over(gripper,{symgoal[1]})"] and state[f'grasped({symgoal[0]})']
            return condition

    return Beta
      



# set up executor classes



def setupExecutors():
    #TODO: Figure out how to populate the executors, should be list of policies that have I, beta, path, etc
    #Populate dictionary with executor name as key, 
    #executors_id_list = ['follow_lane','turn_left','turn_right','change_lane_left','change_lane_right','break','pick_up','drop','park']


    #From brain.py in reload_synapses line 240
    # executors_id_list = domain_synapses.executors_id_list
    # novelty_patterns = domain_synapses.novelty_patterns
    # applicator = domain_synapses.applicator
    # executors = domain_synapses.executors
    pass


def load_infos():
    #TODO: Loads executor data from existing trained polices and novelties
    pass



def select_closest_pattern(novelty_list, operator, same_operator=False, use_base_policies=True, hrl=False):
	pass

def select_best_executor(novelty_list, queue, state, executor_id=None, skill_selection=True):
	pass

def I_in_state(I, state):
	pass
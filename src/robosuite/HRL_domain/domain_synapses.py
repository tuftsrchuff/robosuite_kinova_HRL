import robosuite as suite
from HRL_domain.joint_friction import JointNovelty
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
        env_name="Hanoi", # try with other tasks like "Stack" and "Door"
        robots="Kinova3",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    return env



# set up executor classes
executors = dict()


# domain dependant ids of executors
executors_id_list = ['follow_lane','turn_left','turn_right','change_lane_left','change_lane_right','break','pick_up','drop','park']

# used to select source policy to transfer from whenever the agent needs to train on a novelty pattern
novelty_patterns = {'follow_lane':[],'turn_left':[],'turn_right':[],'break':[],'park':[],'change_lane_left':[],'change_lane_right':[]}

# set up applicator mapping from operators to executors
applicator = {'goForward':['follow_lane'],
		'turnLeft':['turn_left'],
		'turnRight':['turn_right'],
		'ChangeLaneLeft':['change_lane_left'],
		'ChangeLaneRight':['change_lane_right'],
		'Break':['break'],
		'PickUp':['pick_up'],
		'Drop':['drop'],
		'Park':['park']
		}


novelties_info = {
    "joint_friction": {"wrapper":JointNovelty, "params": None, "type":"global"}
}


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

'''
# This files implements the structure of the executor object used to execute the hierarchical policies
'''
from stable_baselines3 import SAC, PPO
import traceback
from robosuite.HRL_domain.detector import Detector

class Executor():
    def __init__(self, env, policy):
      self.policy = policy
      self.env = env
      self.detector = Detector(env)
      
    def execute_policy(self):
        done = False
        rew_eps = 0
        info = None
        step_executor = 0
        model = SAC.load()
        
        #Run executor until it's done, done state can be checked in detector? If block 2 on 3, etc
        MOVE D1 D2 PEG2
        Reach-pick d1
        Pick d1
        reach drop peg2
        drop peg2
        





        done = False
		rew_eps = 0
		info = None
		step_executor = 0

		#Was passed in, now initialized as 0 here
		step_executor = 0
		model = SAC.load(executors[executor].policy, env=self.env, 
                    custom_objects={"observation_space":self.low_env.observation_space, "action_space":self.low_env.action_space})
		exec_env = self.low_env
        
        #Beta needs operator and env passed into it, where are those?
        #Removed and self.Beta() != True and and steps_taken <= step_per_sub
		while not done:
		    action, _states = model.predict(obs)
		    obs, reward, done, info = exec_env.step(action)

            #Why is this different for hrl vs low-level policy?
            step_executor += 1
            rew_eps += reward
            done = executors[executor].Beta(operator=plan[planID], env=self.low_env)

            #or info['collision']
            if step_executor > 1000:
                done = True

        print("Comparing effects")
        # comparing execution effects to expected effects
        new_state = detector(self.low_env)
        expected_effects = effects(plan[planID])

        #Compare looks at all predicates in new state and checks if it exists in grounded
        #predicates in old state, if not adds it to the execution_effects
        execution_effects = new_state.compare(old_state)
        self.verboseprint("The operator expected effects are: {}, the execution effects are: {}.".format(expected_effects, execution_effects))
        success = (all(x in execution_effects for x in expected_effects))
        if success:
            return True
        else:
            self.verboseprint("\n{} failed. Moving to next executor in {} queue.\n".format(executors[executor].id, plan[planID]))
            return False
          
	
	#Executors should be globally defined, in detector function or executor object initialization?
	#executors=None, 
	#executor=None,
	#step_executor=None
	def execute_policy(self,  
					detector=None,
					plan=None,
					planID=0, 
					effects=None, 
					old_state=None, 
					obs=None
					):
		'''
			This function executes the plan on the domain step by step
			### I/P: all domain specific info including dict of executors, executor to execute,
					 detector function to map sub-symb to symbolic, environment object, low level environment (?),
					 list of plan, planID index, effects function for expected effects of plan, and the previous state
			### O/P: True/False with whether policy executed with expected effects
		'''
		done = False
		rew_eps = 0
		info = None
		step_executor = 0

		#Was passed in, now initialized as 0 here
		step_executor = 0
		model = SAC.load(executors[executor].policy, env=self.env, 
                    custom_objects={"observation_space":self.low_env.observation_space, "action_space":self.low_env.action_space})
		exec_env = self.low_env
        
        #Beta needs operator and env passed into it, where are those?
        #Removed and self.Beta() != True and and steps_taken <= step_per_sub
		while not done:
		    action, _states = model.predict(obs)
		    obs, reward, done, info = exec_env.step(action)

            #Why is this different for hrl vs low-level policy?
            step_executor += 1
            rew_eps += reward
            done = executors[executor].Beta(operator=plan[planID], env=self.low_env)

            #or info['collision']
            if step_executor > 1000:
                done = True

        print("Comparing effects")
        # comparing execution effects to expected effects
        new_state = detector(self.low_env)
        expected_effects = effects(plan[planID])

        #Compare looks at all predicates in new state and checks if it exists in grounded
        #predicates in old state, if not adds it to the execution_effects
        execution_effects = new_state.compare(old_state)
        self.verboseprint("The operator expected effects are: {}, the execution effects are: {}.".format(expected_effects, execution_effects))
        success = (all(x in execution_effects for x in expected_effects))
        if success:
            return True
        else:
            self.verboseprint("\n{} failed. Moving to next executor in {} queue.\n".format(executors[executor].id, plan[planID]))
            return False



# Detected objects

cube1_body = env.sim.model.body_name2id('cube1_main')

cube2_body = env.sim.model.body_name2id('cube2_main')

cube3_body = env.sim.model.body_name2id('cube3_main')

peg1_body = env.sim.model.body_name2id('peg1_main')

peg2_body = env.sim.model.body_name2id('peg2_main')

peg3_body = env.sim.model.body_name2id('peg3_main')

obj_body_mapping = {

'o1': cube1_body,

'o2': cube2_body,

'o6': cube3_body,

'o3': peg1_body,

'o4': peg2_body,

'o5': peg3_body

}

obj_mapping = {'o1': 'cube1', 'o2': 'cube2', 'o6': 'cube3', 'o3': 'peg1', 'o4': 'peg2', 'o5': 'peg3'}

area_pos = {'peg1': env.pegs_xy_center[0], 'peg2': env.pegs_xy_center[1], 'peg3': env.pegs_xy_center[2]}


#Symbolic termination for beta
def termination_indicator(operator):

 if operator == 'pick':

def Beta(env, symgoal):

detector = Robosuite_Hanoi_Detector(env)

state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

condition = state[f"grasped({symgoal})"]

return condition

elif operator == 'drop':

def Beta(env, symgoal):

detector = Robosuite_Hanoi_Detector(env)

state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

condition = state[f"on({symgoal[0]},{symgoal[1]})"]

return condition

elif operator == 'reach_pick':

def Beta(env, symgoal):

detector = Robosuite_Hanoi_Detector(env)

state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

condition = state[f"over(gripper,{symgoal})"]

return condition

elif operator == 'reach_drop':

def Beta(env, symgoal):

detector = Robosuite_Hanoi_Detector(env)

state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

condition = state[f"over(gripper,{symgoal})"]

return condition

return Beta




--------------------
reach_pick = Executor_RL(id='ReachPick',

 alg=sac.SAC,

policy="/home/lorangpi/Enigma/data/demo_seed_7/2024-04-30_20:44:23_reach_pick/policy/best_model.zip",

I={},

Beta=termination_indicator('reach_pick'),

nulified_action_indexes=[3],

wrapper = ReachPickWrapper,

horizon=200)

 

success = self.Beta(env, symgoal)
------------
obj_mapping[obj_to_drop]
symgoal = obj_mapping[obj_to_drop]

-----------------
goal = #3d body coordinates of the grounding
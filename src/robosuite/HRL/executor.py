
'''
# This files implements the structure of the executor object used to execute the hierarchical policies
'''
from stable_baselines3 import SAC, PPO
import traceback
from HRL_domain.domain_synapses import *
from robosuite.HRL_domain.detector import Detector

class Executor():
	def __init__(self, 
			  id=1, 
			#   policy=None, 
			  operator = None,
			  I=None, #initiation set which is symb states where action executor is available for execution
			#   Beta=None, 
			#   Circumstance=None, #What is circumstance?
			#   basic=False, #What is basic?
			  hrl=False, 
			  high_env=None,
			  env=None, 
			  low_env=None, ):
		self.id = id
		self.operator = operator
		# self.policy = policy
		self.operator = operator
		self.I = I
		self.hrl = hrl
		self.high_env = high_env
		self.env = env
		self.low_env = low_env
		self.detector = Detector(env)
		populateExecutorInfo(env)


	
	#Executors should be globally defined, in detector function or executor object initialization?
	#executors=None, 
	#executor=None,
	#step_executor=None
	def execute_policy(self,  
					# detector=None,
					plan=None,
					planID=0, 
					symgoal = None
					# effects=None, 
					# old_state=None, 
					# obs=None
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
		# step_per_sub = 10
		# steps_taken = 0
		model = SAC.load(executors[self.operator])

		#Base action generic
		base_action = np.zeros(len(self.env.action_space.sample()))
		obs, _, _, _, _ = self.env.step(base_action)

		#Add goal to observation space if needed
		obs = addGoal(obs, symgoal, self.env, self.operator)
		




		try:
			# executor execution

			#Aren't we already checking if it's hrl with the .hrl property?
			#Why is self.high_env == None necessary?
			#Prev was if executors[executor].hrl
			if self.hrl:
				if self.high_env == None:
					return obs, rew_eps, done, info
				model = PPO.load(executors[self.operator], env=self.env)
				#self.hrl_env.overwrite_executor_id(executor)
				# self.env.overwrite_executor_id(executor)
				exec_env = self.env
				#exec_env.overwrite_executor_id(executor)
				print("HRL EXECUTOR")
			else:
				model = SAC.load(executors[self.operator], env=self.low_env, 
					 custom_objects={"observation_space":self.low_env.observation_space, "action_space":self.low_env.action_space})
				exec_env = self.low_env
			


			Beta = termination_indicator(self.operator)
			terminated = False
			#Beta needs operator and env passed into it, where are those?
			#Removed and self.Beta() != True and and steps_taken <= step_per_sub
			while not done and not terminated:
				action, _states = model.predict(obs)
				obs, reward, terminated, truncated, info = self.env.step(action)
				obs = addGoal(obs, symgoal, self.env, self.operator)

				#Why is this different for hrl vs low-level policy?
				step_executor += 10 if self.hrl else 1
				rew_eps += reward
				done = Beta(self.env, symgoal)
				# done = executors[executor].Beta(operator=plan[planID], env=self.low_env)

				#or info['collision']
				if step_executor > 1000:
					done = True
				steps_taken += 1


			print(f"Terminated: {terminated}")
			print(f"Done: {done}")
			print("Comparing effects")
			# comparing execution effects to expected effects
			new_state = self.detector.get_groundings(self.env)

			# expected_effects = effects(plan[planID])
			expected_effects_keys = effects(self.operator, symgoal)
			execution_effects = []
			for effect in expected_effects_keys:
				execution_effects.append(new_state[effect])
			expected_effects = effect_mapping[self.operator]

			#Compare looks at all predicates in new state and checks if it exists in grounded
			#predicates in old state, if not adds it to the execution_effects
			# execution_effects = new_state.compare(old_state)
			
			success = (all(x in execution_effects for x in expected_effects))
			if success:
				return True
			else:
				print(f"{self.operator} failed...")
				return False
		# exceptions handling
		except FileNotFoundError: 
			print(f"FileNotFound: Tried to execute {self.operator} but it failed. Trying to continue...")
			return False
		except RuntimeError:
			self.verboseprint(f"\nRuntime Error while executing {self.operator}. Moving to next executor in queue.\n")
			success = False
		except Exception as error:
			print(traceback.format_exc())
			print("Error")

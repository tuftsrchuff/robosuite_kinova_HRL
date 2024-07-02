
'''
# This files implements the structure of the executor object used to execute the hierarchical policies

'''
from stable_baselines3 import SAC, PPO
import traceback

class Executor():
	def __init__(self, 
			  id=1, 
			  policy=None, 
			  I=None, 
			  Beta=None, 
			  Circumstance=None, 
			  basic=False, 
			  hrl=False, 
			  verbose=False, 
			  high_env=None,
			  env=None, 
			  low_env=None, ):
		super().__init__()
		self.id = id
		self.policy = policy
		self.I = I
		self.Circumstance = Circumstance
		self.Beta = Beta
		self.basic = basic
		self.hrl = hrl
		self.verbose = verbose
		self.verboseprint = print if verbose else lambda *a, **k: None
		self.high_env = high_env
		self.env = env
		self.low_env = low_env

	def path_to_json(self):
		return {self.id:self.policy}
	
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
		step_per_sub = 10
		steps_taken = 0

		#Was passed in, now initialized as 0 here
		step_executor = 0
		try:
			# executor execution

			#Aren't we already checking if it's hrl with the .hrl property?
			#Why is self.high_env == None necessary?
			if executors[executor].hrl:
				if self.high_env == None:
					return obs, rew_eps, done, info
				model = PPO.load(executors[executor].policy, env=self.env)
				#self.hrl_env.overwrite_executor_id(executor)
				self.env.overwrite_executor_id(executor)
				exec_env = self.env
				#exec_env.overwrite_executor_id(executor)
				print("HRL EXECUTOR")
			else:
				model = SAC.load(executors[executor].policy, env=self.low_env, 
					 custom_objects={"observation_space":self.low_env.observation_space, "action_space":self.low_env.action_space})
				exec_env = self.low_env
			
			#Beta needs operator and env passed into it, where are those?
			#Removed and self.Beta() != True and and steps_taken <= step_per_sub
			#Can 
			while not done:
				action, _states = model.predict(obs)
				obs, reward, done, info = exec_env.step(action)

				#Why is this different for hrl vs low-level policy?
				step_executor += 10 if executors[executor].hrl else 1
				rew_eps += reward
				done = executors[executor].Beta(operator=plan[planID], env=self.low_env)

				#or info['collision']
				if step_executor > 1000:
					done = True
				steps_taken += 1

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
		# exceptions handling
		except FileNotFoundError: 
			self.verboseprint("FileNotFound: Tried to execute {} '{}', but it failed. Trying to continue...".format(executors[executor].id, executor))
			return False
		except RuntimeError:
			self.verboseprint("\nRuntime Error while executing {}. Moving to next executor in {} queue.\n".format(executors[executor].id, plan[planID]))
			success = False
		except Exception as error:
			print(traceback.format_exc())
			print("Error")

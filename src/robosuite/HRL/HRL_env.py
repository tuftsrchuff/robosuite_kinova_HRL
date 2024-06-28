import gym
import numpy as np
from gym import spaces
from domain_synapses import *

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC, PPO
import traceback
import time

#TODO: proper file paths for domain synapses imports

class HRL_FollowLane(gym.Env): # pylint: disable=too-many-instance-attributes
	"""
	Gym environment, where the agent has to control both steering and throttle such that that
	the car stays within the lane and holds the target velocity.
	Car is supposed to follow lane and go left if possible, otherwise go straight.
	If neither straight nor left is possible, car is supposed to stand still.
	"""

	def __init__(self, env, high_env=None, params=None, runtime_settings=None, verbose=False, novelty_pattern=None, executor_id=None):
		# print("HRL Follow initialized...")


		# time.sleep(15)
		self.low_env = env
		self.high_env = high_env
		self.novelty_pattern = novelty_pattern
		self.executor_id = executor_id
		self.step_executor = 0

		self.verbose = verbose
		self.verboseprint = print if verbose else lambda *a, **k: None
		self.verboseprint("\n\n---------------------------- RUNNING AN HRL_FollowLane INSTANCE. --------------------------\n-")

		# TODO: Need to make this a variable passed in
		self.action_space = spaces.Discrete(3)



		self.observation_space = self.low_env.observation_space

		#self.verboseprint("\n\ninit_size obs: \n", self.observation_space)
		self.verboseprint("init_size act: \n", self.action_space)
		print("HRL novelty pattern: ", novelty_pattern)
		print('\n')
		print(f"Action space low env {self.low_env.action_space}")
		time.sleep(5)

	def step(self, action):
		print("In HRL step")
		# print(traceback.print_stack())
		time.sleep(15)

		done = False
		info = {}
		rew_eps = 0
		info.update({"success": False})

		executor_queue = applicator[list(applicator)[action]]
		print(f"Executor queue {executor_queue}")
		#self.verboseprint('The HRL action is the operator: ', list(applicator)[action])
		state = detector(self.low_env)

		#print("HRL EXEC ID: ", self.executor_id)
		executor = select_best_executor(self.novelty_pattern, executor_queue, state, self.executor_id, skill_selection=False)
		print("\nExecutor : {}, specialized on {}.".format(executor, novelty_patterns[executor]))
		self.verboseprint("\nExecutor : {}, specialized on {}.".format(executor, novelty_patterns[executor]))

		#self.verboseprint("\nExecuting operator: {}, selected executor {}.".format(list(applicator)[action], executor))
		#self.verboseprint("Loading policy {}".format(executors[executor].policy))
		#print("\nExecutor : {}, specialized on {}.".format(executor, novelty_patterns[executor]))
		obs = self.low_env.env_obs

		try:
			#print(executors[executor].id)
			if executors[executor].hrl:
				if self.high_env == None:
					return obs, rew_eps, done, info
				model = PPO.load(executors[executor].policy, env=self.high_env)
				#self.hrl_env.overwrite_executor_id(executor)
				self.high_env.overwrite_executor_id(executor)
				exec_env = self.high_env
				#exec_env.overwrite_executor_id(executor)
				print("HRL EXECUTOR")
			else:
				model = SAC.load(executors[executor].policy, env=self.low_env, custom_objects={"observation_space":self.low_env.observation_space, "action_space":self.low_env.action_space})
				exec_env = self.low_env
				print("Low-level executor")
			for i in range(10): # 10 steps per operator
				low_action, _states = model.predict(obs)
				obs, reward, done, info = exec_env.step(low_action)
				rew_eps += reward
				self.step_executor += 10 if executors[executor].hrl else 1
				rew_eps += reward
				#print(info, self.step_executor)
				if self.step_executor > 1000 or info['collision']:
					done = True
				if done:
					break
		except FileNotFoundError: 
			self.verboseprint("FileNotFound: Tried to execute {} '{}', but it failed. Trying to continue...".format(executors[executor].id, executor))
		except RuntimeError:
			self.verboseprint("\nRuntime Error while executing {}. Moving to next executor in {} queue.\n".format(executors[executor].id, list(applicator)[action]))
			success = False
			done = True

		self.obs = obs
		return obs, rew_eps, done, info

	def reset(self):
		self.step_executor = 0
		obs = self.low_env.reset()
		return obs
	
	def close(self):
		self.low_env.close()
	
	def overwrite_executor_id(self, executor_id):
		self.executor_id = executor_id
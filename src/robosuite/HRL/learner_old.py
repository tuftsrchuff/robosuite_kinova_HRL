'''
# This file is the learner of RAPidL.
# This file launches a learning instance, which builds an MDP on the fly and learns to accomodate a novelty, i.e. a performant enough RL policy, with a termination condition
# beta generated from higher level information and transfer knowledge based on some pattern information (here only the label of the novelty, the detection and charaterization of
# the pattern is assumed and is not studied in the paper.)

'''
import copy

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC

from params import *
from planner import *
import domain_synapses
from domain_synapses import *

import gym
import gym_carla_novelty
from gym_carla_novelty.novelties.novelty_wrapper import NoveltyWrapper
from gym_carla_novelty.operator_learners.sb3_eval import FixedSeedEvalCallback

from gym_carla_novelty.operator_learners.train import train

from executor import Executor
from HDDL.generate_hddl import *
from state import *
from carla	import *
from novelties import *

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
set_random_seed(0, using_cuda=True)

# HER parameters
#from stable_baselines3 import HerReplayBuffer, DQN, PPO
#from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
# Available strategies (cf paper): future, final, episode
#goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
# If True the HER transitions will get sampled online
#online_sampling = True

from gym_carla_novelty.operator_learners.training_wrapper import TrainingWrapper
import traceback
import time


class Learner:
	def __init__(self, steps_num, test, seed, transfer, failed_operator, failure_state, novelty_pattern, novelty_id, env_config, settings, eval_settings, verbose, data_folder="", use_basic_policies=True) -> None:
		print("Learner initialized...")
		print(traceback.print_stack())
		# time.sleep(5)
		self.reload_synapses()

		self.test = test
		self.steps_num = steps_num
		self.flag = False

		self.failed_operator = failed_operator.split(" ")[0]
		env_id = training_env[self.failed_operator]
		print("\nCreating eval env.")
		eval_env = gym.make(env_id, runtime_settings=eval_settings, params=env_config, verbose=verbose)

		self.queue_number = '_' + str(len(applicator[self.failed_operator]))

		for predicate in failure_state.grounded_predicates:
			if predicate[0] == "at":
				self.reset_loc = predicate[2]
			elif predicate[0] == "dir":
				self.reset_dir = predicate[1]
		
		print("Creating training env.")
		env = gym.make(env_id, runtime_settings=settings, params=env_config, verbose=verbose)
		env = TrainingWrapper(env)

		if novelty_id is not None:
			if 'obstacle' in novelty_id:
				self.reset_loc = "l2"
				self.reset_dir = "e"
				self.flag = True
				# Need to train on the whole state space
				# for HER or in case we want to terminate the training earlier (when reaching a high level goal)
				#env = NoveltyWrapper(env, loc="l2", direction="s", goal_function=self.DesiredGoal_generator)
				#eval_env = NoveltyWrapper(eval_env, loc="l2", direction="s", goal_function=self.DesiredGoal_generator)
				print("Training from fixed initial state: reset_loc={}, reset_dir={}".format(self.reset_loc, self.reset_dir))
				env = NoveltyWrapper(env, loc=self.reset_loc, direction=self.reset_dir)
				eval_env = NoveltyWrapper(eval_env, loc=self.reset_loc, direction=self.reset_dir)
				self.training_steps = int(self.steps_num/3)
				self.eval_freq = int(self.steps_num/(3*5))

			elif len(novelty_id) > 1 or novelties_info[novelty_id[0]]["type"] == "global":
				# Need to train on the whole state space
				# for HER or in case we want to terminate the training earlier (when reaching a high level goal)
				#env = NoveltyWrapper(env, loc="l2", direction="s", goal_function=self.DesiredGoal_generator)
				#eval_env = NoveltyWrapper(eval_env, loc="l2", direction="s", goal_function=self.DesiredGoal_generator)
				print("Training from random initial state: reset_loc={}, reset_dir={}".format(self.reset_loc, self.reset_dir))
				env = NoveltyWrapper(env)
				eval_env = NoveltyWrapper(eval_env)
				self.training_steps = int(self.steps_num/3)
				self.eval_freq = int(self.steps_num/(3*5))
			elif novelties_info[novelty_id[0]]["type"] == "local":
				# Need to train only from the failure state
				#desired_effects = effects(self.failed_operator)
				#desired_effects = self.DesiredGoal_generator(self.failure_env, self.failed_operator, state=failure_state)
				#env = NoveltyWrapper(env, loc=self.reset_loc, direction=self.reset_dir, high_goal=desired_effects)
				#eval_env = NoveltyWrapper(eval_env, loc=self.reset_loc, direction=self.reset_dir, high_goal=desired_effects)
				#eval_env = NoveltyWrapper(eval_env, loc="l2", direction="s", high_goal=desired_effects)
				env = NoveltyWrapper(env, loc=self.reset_loc, direction=self.reset_dir)
				eval_env = NoveltyWrapper(eval_env, loc="l2", direction="s")
				self.training_steps = self.steps_num * 2
				self.eval_freq = int(self.steps_num * 2/5)
				#print("desired effects: ", desired_effects)
				print("Starting learning, reseting on reset_loc={}, and reset_dir={}".format(self.reset_loc, self.reset_dir))

		if novelty_id is not None:
			novelties = []
			novelties_eval = []
			for novelty in novelty_id:
				print("\033[1m" + "\n\n\t===> Injecting Novelty: {} <===\n\n".format(novelty) + "\033[0m")
				novel_wrapper = novelties_info[novelty]["wrapper"]
				novel_params = novelties_info[novelty]["params"]
				if novel_params is None:
					novel_params = {}
				novelties.append(novel_wrapper(env=env, **novel_params))
				novelties_eval.append(novel_wrapper(env=eval_env, **novel_params))
			env.set_novelties(novelties)
			eval_env.set_novelties(novelties_eval)
		
		env.seed(seed)
		eval_env.seed(seed)

		self.env = env
		self.eval_env = eval_env

		self.name = failed_operator.split(' ')[0] + self.queue_number
		self.folder = data_folder + self.name + '/'
		self.policy_folder = self.folder + "policy/" + self.name
		self.tensorboard_folder = self.folder+"tensorboard/"


		# Source policy transfer strategy
		print("faced novelty pattern = ", novelty_pattern)
		source_policy = self.select_source_policy(novelty_pattern, transfer, use_basic_policies)
		self.learned = self.learn_policy(source_policy)
		if self.learned:
			self.abstract_to_executor()	
			novelty_patterns.update({self.name:novelty_pattern})
		self.env.close()
		print("Closing training env.")
		try:
			self.eval_env.close()
			print("Closing eval env.")
		except:
			pass

	def reload_synapses(self):
		global applicator 
		global executors
		global novelty_patterns
		global executors_id_list

		executors_id_list = domain_synapses.executors_id_list
		novelty_patterns = domain_synapses.novelty_patterns
		applicator = domain_synapses.applicator
		executors = domain_synapses.executors

	def DesiredGoal_generator(self, env, operator, state=None):
		# Use planner to map state to desired effects knowing the target operator to train on
		if state == None:
			state = detector(env)
		generate_hddls(state, task=operator, filename="beta_problem")
		plan, game_action_set = call_planner("beta_domain", "beta_problem") # get a plan
		if plan==False or game_action_set==False:
			return False
		desired_goal = effects(plan[0])
		return desired_goal

	def select_source_policy(self, novelty_pattern, transfer, use_basic_policies):
		# Handling policy source RL transfer - Source policy transfer strategy
		if transfer and novelty_pattern is not None: # if the agent charaterizes some information about the novelty
			"""
			if len(novelty_pattern) > 1:
				# if many novelties detected, then transfer from the last policy that learned to execute the failed operator
				return executors[applicator[self.failed_operator][-1]].policy
			elif novelties_info[novelty_pattern[0]]["type"] == "global":
				# tranfer from a policy that accomodated a novelty with a similar pattern as the current novelty
				source = select_closest_pattern(novelty_pattern, self.failed_operator.split(' ')[0])
				if source != None:
					return executors[source].policy 
				# if the agent has not yet faced such novelty pattern, then transfer from the last policy that learned to execute the failed operator
				return executors[applicator[self.failed_operator][-1]].policy
			elif novelties_info[novelty_pattern[0]]["type"] == "local":
				# transfer from the last policy that learned to execute the failed operator
				return executors[applicator[self.failed_operator][-1]].policy
			"""
			source = select_closest_pattern(novelty_pattern, self.failed_operator.split(' ')[0], same_operator=True, use_base_policies=use_basic_policies)
			if source != None:
				print("Transferring from source policy: {}, trained on {}.".format(source, novelty_patterns[source]))
				print("Path to source model: {}".format(executors[source].policy))
				return executors[source].policy 
		return None

	def learn_policy(self, source_policy):
		if source_policy == None:
			policy_kwargs = dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]))
			model = SAC("MlpPolicy", self.env, verbose=1, gamma=0.95, learning_rate=0.0003, policy_kwargs=policy_kwargs, tensorboard_log=self.tensorboard_folder+self.name, device="cuda")
		else:
			model = SAC.load(source_policy, env=self.env, tensorboard_log=self.tensorboard_folder, device="cuda") 
			model.set_env(self.env)
			parameters = model.get_parameters()
			model.set_parameters(parameters)

		policy_kwargs = dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]))
		if self.test:
			model = train(
				self.env, 
				eval_env=self.eval_env, 
				model=model, 
				policy_kwargs=policy_kwargs, 
				reward_threshold=810,
				save_freq=20, 
				total_timesteps=30,
				best_model_save_path=self.policy_folder,
				eval_freq=20, 
				n_eval_episodes=1,
				save_path=self.folder,
				run_id=self.name)
		else:
			model = train(
				self.env, 
				eval_env=self.eval_env, 
				model=model, 
				policy_kwargs=policy_kwargs, 
				reward_threshold=810,
				save_freq=self.eval_freq, 
				total_timesteps=self.training_steps,
				best_model_save_path=self.policy_folder,
				eval_freq=self.eval_freq, 
				n_eval_episodes=20,
				save_path=self.folder,
				run_id=self.name)

		self.model = model
		#self.model.save(f"{self.folder}models/{self.name}")
		return True

	def abstract_to_executor(self):
		#self.model.save("gym_carla_novelty/policies/" + self.name)
		applicator[self.failed_operator].append(self.name)
		#executor = Executor(id=self.name, policy=f"{self.folder}models/{self.name}", Beta=beta_indicator) 
		if self.flag:
			executor = Executor(id=self.name, policy=f"{self.policy_folder}/best_model", Beta=beta_indicator, I=[[["at","car","l2"],["dir","e"]],[["at","car","l3"],["dir","e"]]])
		else:
			executor = Executor(id=self.name, policy=f"{self.policy_folder}/best_model", Beta=beta_indicator)
		ex_dict = {self.name:executor}
		executors.update(ex_dict)
		executors_id_list.append(self.name)



	"""		
	# in the init function
	def beta_indicator(desired_effects):
		# use planner to map initial state to desired effects
		def success_func(env):
			from state import State
			state = detector(env)
			return desired_effects in state.grounded_predicates
		return success_func"""

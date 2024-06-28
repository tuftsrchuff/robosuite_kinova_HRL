'''
# This file is the Brain of RAPidL.
# It is the main file which talks to the planner, learner and the game instance.

Important References

'''
from __future__ import print_function
import copy
import sys
import json
import os
from datetime import datetime

import gym
import gym_carla_novelty
import domain_synapses
from importlib import reload
from gym_carla_novelty.novelties.novelty_wrapper import NoveltyWrapper
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.utils import set_random_seed

from HDDL.generate_hddl import *
from planner import *
from learner import Learner
from learner_HRL import Learner_HRL
from domain_synapses import *
from novelties import *
from executor import Executor
import time

import carla
import traceback
import time


#eval_runtime_settings = {'address': ["localhost", 2002],'no_rendering': True, 'client': eval_client, 'reload_world': False}

set_random_seed(0, using_cuda=True)
# Env parameters
env_config = {'discrete': False, 'stop_prob':0.0, 'max_offset': 4, 'ignore_stops': True, 'change_stop_prob':0}

class Brain:
	'''        
		Brain is the main RapidLearn class.
		It manages planning, execution and learning modules.
		Brain also structures the skills together with domain dependent variables present in domain_synapses.py. 
		It is able to store all learned skills, and to recover them if needed even after quiting a run.
	'''
	def __init__(self, steps_num, client, eval_client, client_port, eval_client_port, domain="domain_example", verbose=False, DATA_DIR="", transfer=True, hrl=False, seed=0, test=False, use_base_policies=True):
		self.client = client
		self.eval_client = eval_client
		self.client_port = client_port
		self.eval_client_port = client_port + eval_client_port
		self.domain = domain
		self.learner = None
		self.steps_num = steps_num
		self.learned_policies_dict = {'goForward':{},'turnLeft':{},'turnRight':{},'ChangeLaneLeft':{},'ChangeLaneRight':{}} # store failed action:learner_instance object
		self.task_episode = {}
		self.verbose = verbose
		self.verboseprint = print if verbose else lambda *a, **k: None
		self.DATA_DIR = DATA_DIR
		self.transfer = transfer
		self.hrl = hrl
		self.seed = seed
		self.use_base_policies = use_base_policies
		self.env_id = None
		self.env = None
		self.test = test
		self.novelty_list = None
		# print("Brain initialized...")
		# print(traceback.print_stack())
		# time.sleep(5)


	def generate_env(self, env_id, novelty_list=None, reset_loc=None, reset_dir=None, render=False, only_train=False):
		'''        
			Creates the training and evaluation gym environments depicting the conditions in novelty_list.
			If reset_dir and reset_loc are not None, the environment will always reset from the given reset state.
		'''
		print("Generating environment...")
		print(f"Env ID: {env_id} \nNovelty List {novelty_list} \nReset Loc {reset_loc} \nReset Dir: {reset_dir}\nRender: {render}\nTrain: {only_train}")
		time.sleep(5)
		if novelty_list == []:
			novelty_list = None

		if self.env == None or novelty_list != self.novelty_list:
			
			if self.env != None:
				self.close_env()

			self.render = render
			self.reset_loc = reset_loc
			self.reset_dir = reset_dir
			self.env_id = env_id

			# How the environment is affected by the novelties either "local" or "global". If multiple novelties, the locality is global anyway
			if novelty_list == None:
				self.locality = None
			else:
				if len(novelty_list) > 1 or novelties_info[novelty_list[0]]["type"] == "global":
					self.locality = "global"
				elif novelties_info[novelty_list[0]]["type"] == "local":
					self.locality = "local"

			# set to False, if you wish to watch the agent train in the viewport (very slow!):
			self.runtime_setting = {'address': ['localhost', self.client_port], 'no_rendering': not(render), 'client': self.client, 'reload_world': False}
			self.eval_runtime_settings = {'address': ['localhost', self.eval_client_port], 'no_rendering': not(render), 'client': self.eval_client, 'reload_world': False}

			if not(only_train):

				print("\nCreating execution env.")
				print(f"Novelty List: {novelty_list}")
				# get environment instances before injecting novelty
				self.verboseprint("env id is: ",env_id)
				env = gym.make(env_id, runtime_settings=self.runtime_setting, params=env_config, verbose=self.verbose) # make a new instance of the environment. # obs = env.reset(loc="l2", direction="s") 

				#TODO dictionary for each observation for every semantic.
				#pre_observation_semantic = copy.deepcopy(env.observation_semantic)
				#pre_actions_semantic = copy.deepcopy(env.actions_semantic)
				if reset_loc == None:
					env = NoveltyWrapper(env)
				else:
					env = NoveltyWrapper(env, loc=reset_loc, direction=reset_dir)

				# Injecting the novelties in the environment
				if novelty_list is not None:
					if type(novelty_list) == str:
						novelty_list = [novelty_list]
					novelties = []
					for novelty in novelty_list:
						self.verboseprint("\033[1m" + "\n\n\t===> Injecting Novelty: {} <===\n\n".format(novelty) + "\033[0m")
						novel_wrapper = novelties_info[novelty]["wrapper"]
						novel_params = novelties_info[novelty]["params"]
						if novel_params is None:
							novel_params = {}
						novelties.append(novel_wrapper(env=env, **novel_params))
					env.set_novelties(novelties)
				env.seed(self.seed)
				self.env = env
				self.loop_env = gym.make("carla-follow-lane-v1", env=env, verbose=False, novelty_pattern=novelty_list)
				self.loop_env.reset()
				self.hrl_env = gym.make("carla-follow-lane-v1", env=env, high_env=self.loop_env, verbose=False, novelty_pattern=novelty_list)
				self.hrl_env.reset()
		self.novelty_list = novelty_list

	def close_env(self):
		'''        
			Closes the execution gym environment.
		'''
		if self.env != None:
			self.env.close()
			self.env = None
			print("Closing execution env.")

	def save_infos(self):
		'''        
			Saves important information for brain checkpoint into jsons resuming learned skills.
			Saves the executors, the mapping function from operators to sets of executors, and information about the novelty patterns that have already been trained on.
		'''
		policies_paths = {}
		for operator, executors_set in applicator.items():
			for executor in executors_set:
				policies_paths.update(executors[executor].path_to_json())
		j = json.dumps(applicator, indent=4)
		with open(self.DATA_DIR+'applicator.json', 'w') as f:
			print(j, file=f)
		j = json.dumps(novelty_patterns, indent=4)
		with open(self.DATA_DIR+'novelty_patterns.json', 'w') as f:
			print(j, file=f)
		j = json.dumps(policies_paths, indent=4)
		with open(self.DATA_DIR+'policies_paths.json', 'w') as f:
			print(j, file=f)
		j = json.dumps(executors_id_list, indent=4)
		with open(self.DATA_DIR+'executors_id_list.json', 'w') as f:
			print(j, file=f)
		j = json.dumps(self.learned_policies_dict, indent=4)
		with open(self.DATA_DIR+'learned_policies.json', 'w') as f:
			print(j, file=f)

		main_folder, _ = os.path.split(self.DATA_DIR[:-1])
		if not(main_folder.endswith('/')):
			main_folder += '/'
		j = json.dumps(applicator, indent=4)
		with open(main_folder+'applicator.json', 'w') as f:
			print(j, file=f)
		j = json.dumps(novelty_patterns, indent=4)
		with open(main_folder+'novelty_patterns.json', 'w') as f:
			print(j, file=f)
		j = json.dumps(policies_paths, indent=4)
		with open(main_folder+'policies_paths.json', 'w') as f:
			print(j, file=f)
		j = json.dumps(executors_id_list, indent=4)
		with open(main_folder+'executors_id_list.json', 'w') as f:
			print(j, file=f)
		j = json.dumps(self.learned_policies_dict, indent=4)
		with open(main_folder+'learned_policies.json', 'w') as f:
			print(j, file=f)

	def load_infos(self):
		'''        
			Loads saved information from a data folder for brain recovery (a form of data checkpoint for the brain to recover learned skills)
			Loads the executors, the mapping function from operators to sets of executors, and information about the novelty patterns that have already been trained on.
		'''
		self.verboseprint('\n_____________________________________________________________________________________________________________________')
		self.verboseprint('\n\n\t\t\t\t RELOADING SYNAPSES FROM PREVIOUS EXPERIMENT :')
		main_folder, _ = os.path.split(self.DATA_DIR[:-1])
		if not(main_folder.endswith('/')):
			main_folder += '/'
		self.verboseprint('\t\t\t\t',main_folder,'\n\n')
		f = open(main_folder+'applicator.json')
		domain_synapses.applicator = json.load(f)
		f = open(main_folder+'novelty_patterns.json')
		domain_synapses.novelty_patterns = json.load(f) 
		f = open(main_folder+'policies_paths.json')
		policies_paths = json.load(f)
		f = open(main_folder+'executors_id_list.json')
		domain_synapses.executors_id_list = json.load(f)
		f = open(main_folder+'learned_policies.json')
		self.learned_policies_dict = json.load(f)

		self.reload_synapses()

		for exec_id in domain_synapses.executors_id_list:
			print("Loading executor: ", policies_paths[exec_id])
			try:
				if 'obstacle' in novelty_patterns[exec_id]:
					I = [[["at","car","l2"],["dir","e"]],[["at","car","l3"],["dir","e"]]]
					executor = Executor(id=exec_id, policy=policies_paths[exec_id], Beta=beta_indicator, basic=True, hrl=True, I=I)
					print(exec_id +" is an hrl policy.")
				elif 'traffic' in novelty_patterns[exec_id]:
					executor = Executor(id=exec_id, policy=policies_paths[exec_id], Beta=beta_indicator, basic=True, hrl=True)
					print(exec_id +" is an hrl policy.")			
				else:
					executor = Executor(id=exec_id, policy=policies_paths[exec_id], Beta=beta_indicator, basic=True)
			except KeyError:
				executor = Executor(id=exec_id, policy=policies_paths[exec_id], Beta=beta_indicator, basic=True)
			ex_dict = {exec_id:executor}
			domain_synapses.executors.update(ex_dict)
		
		self.reload_synapses()

		self.verboseprint("\n\nLoading executors_id_list information.\n", executors_id_list)
		self.verboseprint("\n\nLoading novelty_patterns information.\n", novelty_patterns)
		self.verboseprint("\n\nLoading applicator information.\n", applicator)
		self.verboseprint("\n\nLoading executors information.\n", executors)
		self.verboseprint("\n\nLoading learned_policies information.\n", self.learned_policies_dict)
		self.verboseprint('_____________________________________________________________________________________________________________________\n')

	def reload_synapses(self):
		'''        
			Reloads some major domain dependent variables from domain_synapses.py
		'''
		global applicator 
		global executors
		global novelty_patterns
		global executors_id_list

		executors_id_list = domain_synapses.executors_id_list
		novelty_patterns = domain_synapses.novelty_patterns
		applicator = domain_synapses.applicator
		executors = domain_synapses.executors

	def run_brain(self, task, trial=1, max_trials=3, only_eval=False, only_train=False, direct_train=False):
		'''        
			This is the driving function of this class.
			Call the environment and run the environment for x number of trials.
			task must be in the form of "Navigation li " + goal_location
		'''
		direct_training = {'direct':direct_train, 'failed_operator':'goForward l16 l21 s e', 'failure_state':State(init_predicates=[["at","car","l2"],["dir","s"]])}
		novelty_pattern_name = '' if self.novelty_list == None  else '_'.join(sorted(self.novelty_list))
		done = False
		flag = False
		if only_train:
			self.close_env()
			for operator in adaptive_op:
				if self.novelty_list != None:
					if novelties_info[self.novelty_list[0]]["type"] == "local":
						operator = adaptive_op[0]
						flag = True
				if novelty_pattern_name not in self.learned_policies_dict[operator].keys() or flag:
					direct_training["direct"] = True
					direct_training["failed_operator"] = operator
		else:
			if not(direct_training["direct"]):
				self.verboseprint("Agent launching trial {} on task {}.".format(trial, task))
				try:
					self.task_episode[task] += 1
				except:
					self.task_episode.update({task: 0})
					
				if self.env == None:
					if self.env_id == None:
						print("Neither env, nor env_id specified for run_brain trial ", trial)
						sys.exit()
					else:
						self.generate_env(env_id=self.env_id, novelty_list=self.novelty_list, reset_loc=self.reset_loc, reset_dir=self.reset_dir, render=self.render)
				init_dir, init_loc = None, None
				
				while init_loc==None or init_dir==None or init_loc[0]!="l":
					obs = self.env.reset()
					state = detector(self.env)
					self.current_state = state

					for predicate in state.grounded_predicates:
						if predicate[0] == "at":
							init_loc = predicate[2]
						elif predicate[0] == "dir":
							init_dir = predicate[1]

				self.verboseprint("\nInit state: loc = {}, direction = {}".format(init_loc, init_dir))

				generate_hddls(state, task=task)
				self.verboseprint("\033[1m" + "\n\n\tPlanning ...\n\n" + "\033[0m")
				plan, game_action_set = call_planner(self.domain, "problem") # get a plan
				if plan==False or game_action_set==False:
					self.verboseprint("\033[1m" + "\n\n\tAgent couldn't find a plan for: {}, on trial {}.\n\n".format(task, trial) + "\033[0m")
					return done, trial
				done, failed_operator, failure_state = self.execute_plan(self.env, game_action_set, plan, obs)
		if done and not(direct_training['direct']):
			self.verboseprint("\033[1m" + "\n\n\tAgent successfully achieved task: {}, on trial {}.".format(task, trial) + "\033[0m")
			return done, trial # agent is successful in achieving the task
		elif not(only_eval): # Enters recovery mode
			#TODO re-plan without the failed operator to find if another path exist. Our experiments are designed so that re-planning would fail anyway.
			if direct_training["direct"]:
				print("\nWarning: direct_training is set to True, no execution in this run.\n")
				failed_operator = direct_training["failed_operator"]
				failure_state = direct_training["failure_state"]
			# When re-planning still fails: launch learning method
			if trial == 1: # cases when the plan and re-plan failed for the first time and the agent needs to learn a new action using RL
				if self.novelty_list == None:
					novelty_pattern = []
					novelty_pattern_name = ''
				else:
					novelty_pattern = [novelties_info[n]["pattern"] for n in self.novelty_list]
					novelty_pattern_name = '_'.join(sorted(self.novelty_list))
				if novelty_pattern_name not in self.learned_policies_dict[failed_operator.split(' ')[0]].keys():
					# if no policy has yet been trained on the novelty pattern
					self.verboseprint("\033[1m" + "\n\n\tInstantiating a RL Learner to learn a new {} action to solve the impasse, and learn a new executor for {}.".format(self.locality, failed_operator.split(" ")[0]) 
					+ "\n\tThe detected novelty patterns are {} associated with this ids {}.".format(novelty_pattern, self.novelty_list)
					+ "\n\tcurrent time: {}\n\n".format(datetime.now()) + "\033[0m")
					self.learned = self.call_learner(failed_operator=failed_operator, failure_state=failure_state, novelty_pattern=novelty_pattern, verbose=self.verbose)
					if self.learned: # when the agent successfully learns a new action, it should now test it to re-run the environment.
						self.verboseprint("Agent succesfully learned a new action in the form of policy. Now resetting to test.")
						while (not done) and (trial<max_trials):
							done, trial = self.run_brain(task=task, trial=trial+1, only_eval=True)
					else:
						self.verboseprint("Agent failed to learn a new action in the form of policy. Exit..")
						return done, trial # Agent was unable to learn an executor that overcomes the novelty
				else:
					self.verboseprint("Agent failed to execute a policy already trained on the novelty pattern.")
					return done, trial # Agent was unable to learn an executor that overcomes the novelty
			else:
				return done, trial # for trials >1, returns the result of using the newly learned operator in the global task
			if done:
				self.verboseprint("Success!")
			else:
				self.verboseprint("Agent was unable to achieve the task: {}, despite {} trials.".format(task, trial))
			return done, trial # back to main, returns the global results of run_nrain function
		else: 
			return done, trial

	def call_learner(self, failed_operator, failure_state, novelty_pattern=None, verbose=False):
		'''        
			This function instantiates a RL learner to start finding interesting states to send to the planner
			Learner computes Beta from the operator expected effects, and I from the failure state
			it then learns a policy that finds a path from states validated by I, to states validated by Beta
		'''
		print(f"Failed Operator {failed_operator}\nFailure State {failure_state}\nNovelty_pattern {novelty_pattern}\nVerbose {verbose}")
		print(f"HRL {self.hrl}")
		# time.sleep(5)
		if 'obstacle' or 'traffic' in novelty_pattern and self.hrl:
			self.learner = Learner_HRL(steps_num=self.steps_num, test=self.test, seed=self.seed, transfer=self.transfer, failed_operator=failed_operator, failure_state=failure_state, novelty_pattern=novelty_pattern, novelty_id=self.novelty_list, env_config=env_config, settings=self.runtime_setting, eval_settings=self.eval_runtime_settings, verbose=verbose, data_folder=self.DATA_DIR, use_basic_policies=self.use_base_policies) # learns the policiy, I, Beta and C. Stores result as an executor object and maps the executor to the operator's list
		else:
			self.learner = Learner(steps_num=self.steps_num, test=self.test, seed=self.seed, transfer=self.transfer, failed_operator=failed_operator, failure_state=failure_state, novelty_pattern=novelty_pattern, novelty_id=self.novelty_list, env_config=env_config, settings=self.runtime_setting, eval_settings=self.eval_runtime_settings, verbose=verbose, data_folder=self.DATA_DIR, use_basic_policies=self.use_base_policies) # learns the policiy, I, Beta and C. Stores result as an executor object and maps the executor to the operator's list
		novelty_pattern_name = '' if self.novelty_list == None  else '_'.join(sorted(self.novelty_list))
		self.learned_policies_dict[failed_operator.split(' ')[0]].update({novelty_pattern_name: self.learner.learned}) # save the learner instance object to the learned policies dict.
		return self.learner.learned

	def execute_plan(self, env, sub_plan, plan, obs):
		'''
			This function executes the plan on the domain step by step
			### I/P: environment instance and sequence of actions step by step
			### O/P: SUCCESS/ FAIL with the failed action
		'''
		self.verboseprint("Running plan execution.")
		# print(f"Novelty Patterns {novelty_patterns}\n")
		# print(f"Sub plan {sub_plan}\n")
		# print(f"Plan {plan}")
		rew_eps = 0
		self.sub_plan = sub_plan
		self.plan = plan
		i = 0

		#Normal env is hierarchical layer
		env = self.hrl_env

		#Control layer
		low_env = self.hrl_env.low_env
		
		self.verboseprint(sub_plan)
		while (i < len(plan)): # looping through the plan (operator by operator)
			done = False
			success = True
			old_state = detector(low_env)
			self.current_state = old_state
			queue = copy.deepcopy(sub_plan[i])
			print(f"Queue is: {queue}")
			# time.sleep(15)
			if not(self.use_base_policies):
				for executor in queue:
					if executors[executor].basic:
						queue.remove(executor)
			self.verboseprint("\n{}   {}\n".format(i, queue))
			while (len(queue) > 0): # looping through all executors mapped to operator i
				step_executor = 0
				executor = select_best_executor(self.novelty_list, queue, old_state) # selection strategy
				#All executors specialized on empty array
				self.verboseprint("\nExecutor : {}, specialized on {}.".format(executor, novelty_patterns[executor]))
				queue.remove(executor)

				self.verboseprint("\nExecuting plan_step: {}, mapped to executor {}.".format(plan[i], executor))
				self.verboseprint("Loading policy {}".format(executors[executor].policy))

				#executors are global dictionary and defined within reload_synapses, taken from domain synapses
				#Return whether or not the executor was successful or not

				print("Executing policy through the executor...")
				executor_core = Executor(verbose=self.verbose, Beta=beta_indicator)
				try:
					success = executor_core.execute_policy(executors=executors, executor=executor, detector=detector, env=env, 
									  low_env=low_env, plan=plan, planID=i, effects=effects, old_state=old_state, obs=obs,
									  step_executor=step_executor)
				except FileNotFoundError: 
					self.verboseprint("FileNotFound: Tried to execute {} '{}', but it failed. Trying to continue...".format(executors[executor].id, executor))
				except RuntimeError:
					self.verboseprint("\nRuntime Error while executing {}. Moving to next executor in {} queue.\n".format(executors[executor].id, plan[i]))
					success = False
				if success:
					#If any of the expected executor effects exactly match the execution effects break
					break
				

				# try:
				# 	# executor execution
				# 	if executors[executor].hrl:
				# 		model = PPO.load(executors[executor].policy, env=env)
				# 		#self.hrl_env.overwrite_executor_id(executor)
				# 		env.overwrite_executor_id(executor)
				# 		exec_env = env
				# 		#exec_env.overwrite_executor_id(executor)
				# 		print("HRL EXECUTOR")
				# 	else:
				# 		model = SAC.load(executors[executor].policy, env=low_env, custom_objects={"observation_space":low_env.observation_space, "action_space":low_env.action_space})
				# 		exec_env = low_env
				# 	while not done:
				# 		action, _states = model.predict(obs)
				# 		obs, reward, done, info = exec_env.step(action)
				# 		step_executor += 10 if executors[executor].hrl else 1
				# 		rew_eps += reward
				# 		done = executors[executor].Beta(operator=plan[i], env=low_env)
				# 		if step_executor > 1000 or info['collision']:
				# 			done = True

				# 	# comparing execution effects to expected effects
				# 	new_state = detector(low_env)
				# 	expected_effects = effects(plan[i])
				# 	execution_effects = new_state.compare(old_state)
				# 	self.verboseprint("The operator expected effects are: {}, the execution effects are: {}.".format(expected_effects, execution_effects))
				# 	#Operand expected effects match the execution effects

				# 	success = (all(x in execution_effects for x in expected_effects))
				# 	if success:
				# 		break
				# 	else:
				# 		self.verboseprint("\n{} failed. Moving to next executor in {} queue.\n".format(executors[executor].id, plan[i]))
				# # exceptions handling
				# except FileNotFoundError: 
				# 	self.verboseprint("FileNotFound: Tried to execute {} '{}', but it failed. Trying to continue...".format(executors[executor].id, executor))
				# except RuntimeError:
				# 	self.verboseprint("\nRuntime Error while executing {}. Moving to next executor in {} queue.\n".format(executors[executor].id, plan[i]))
				# 	success = False

			if not success:
				self.verboseprint("\nThe execution effects don't match the operator {}. Launching recovery mode.\n".format(plan[i]))
				return False, plan[i], old_state
			i+=1
		return True, None, None







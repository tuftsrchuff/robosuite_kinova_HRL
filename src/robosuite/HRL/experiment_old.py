
import time
from datetime import datetime
import os
import csv
import argparse
from argparse import ArgumentParser
import copy
import uuid
from abc import abstractmethod, ABC
import gym
import random
import sys
import json

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC

from brain import *
from novelties import novelties_info

from gym_carla_novelty.novelties.novelty_wrapper import NoveltyWrapper
from gym_carla_novelty.operator_learners.train import train
from gym_carla_novelty.operator_learners.evaluate_baseline import eval_success
import carla
from domain_synapses import *

def conf_carla(port):
	print("_ Carla using port ", port)
	client = carla.Client("localhost", port)
	client.set_timeout(30)
	world = client.get_world()
	settings = world.get_settings()
	settings.synchronous_mode = True
	settings.no_rendering_mode = True
	settings.fixed_delta_seconds = 0.1
	world.apply_settings(settings)
	world.tick()
	client.load_world("Town07", reset_settings=False)
	return client

class Experiment:
	
	def __init__(self, args, novelty_list, extra_run_ids=''):
		self.DATA_DIR = args['log_storage_folder']
		self.base_port = args['carla_base_port']
		self.eval_port = args['carla_eval_port']
		self.eval_post = args['eval']
		self.eval_pre_post = args['eval_pre_post']
		self.hashid = uuid.uuid4().hex
		self.experiment_id = extra_run_ids
		self.results_dir = self._get_results_dir()
		os.makedirs(self.results_dir, exist_ok=True)
		self.load_brain = args['load']
		self.recover = args['recover']
		expe_path = os.path.split(self.results_dir[:-1])[0]
		# save args
		exclude_keys = ['client', 'eval_client']
		save_args = {k: args[k] for k in set(list(args.keys())) - set(exclude_keys)}
		if not(expe_path.endswith('/')):
			expe_path += '/'
		with open(expe_path+'args.txt', 'w') as f:
			json.dump(save_args, f, indent=2)
		
		self.novelty_list = novelty_list
		self.test = args["test"]
		self.fixed_locations = args["fixed_locations"]
		if self.fixed_locations:
			self.evaluations_num = 20
		else:
			self.evaluations_num = args['trials_eval']
		if self.test:
			self.trials_eval_pre = 5
			self.trials_eval_post = 5
			self.trials_training = 4
		else:
			self.trials_eval_pre = self.evaluations_num
			self.trials_eval_post = self.evaluations_num
			self.trials_training = args['trials_training']
		self.render = args['render']
		self.env_id="carla-executor-env-v0"
		self.verbose = args['verbose']
		self.transfer = args['transfer']
		self.hrl = args['hrl']
		self.seed = args['seed']
		self.steps_num = args['steps']
		self.direct_training = args["direct_training"]
		if self.direct_training:
			self.trials_eval_pre = 0
			self.trials_eval_post = 0

	def _get_results_dir(self):
		if self.experiment_id == '':
			return self.DATA_DIR + os.sep
		return self.DATA_DIR + os.sep + self.experiment_id + os.sep

	@abstractmethod
	def run(self):
		pass

	def write_row_to_results(self, data, tag):
		db_file_name = self.results_dir + os.sep + str(tag) + "results.csv"
		print(db_file_name)
		time.sleep(10)
		with open(db_file_name, 'a') as f:  # append to the file created
			writer = csv.writer(f)
			writer.writerow(data)


class RapidExperiment(Experiment):
	HEADER_TRAIN = ['Episode', 'Done']
	HEADER_TEST = ['Novel','success_rate'] # Novel: 0=pre-novelty_domain, 1=post-novelty_domain

	def __init__(self, args, novelty_list=None, experiment_id='no_id', brain=None, eval_only=False, eval_only_pre_post=False):
		if novelty_list == None:
			novelty_list = args['novelty_list']
		if experiment_id == None:
			experiment_id = '_'.join(novelty_list)
		super(RapidExperiment, self).__init__(args, novelty_list, experiment_id)

		self.write_row_to_results(self.HEADER_TRAIN, "train")
		self.write_row_to_results(self.HEADER_TEST, "test")

		if eval_only_pre_post or self.eval_pre_post:
			self.trials_training = 0
			self.trials_eval_post = 0
		elif eval_only or self.eval_post:
			self.trials_training = 0
			self.trials_eval_pre = 0

		self.novelty = novelty_list
		if type(self.novelty_list) == str:
			self.novelty_list = [self.novelty_list]
		self.brain = brain

		self.client = args['client']
		self.eval_client = args['eval_client']

		print('General experiment directory is:', os.path.split(self.results_dir[:-1])[0])
		print('Novelty result directory is:', self.results_dir)

	def run(self):
		print("\033[1m" + "\n\n\t\t\t\t\t===> RAPIDLEARN EXPERIMENT ON: {} <===\n\n".format(self.novelty) + "\033[0m")
		if self.hrl and 'obstacle' in self.novelty:
			print("\033[1m" + "\t\t\t\t\t===> THE EXPERIMENT USES HIERARCHICAL ACTION ABSTRACTION ON LOCAL NOVELTIES <===\n\n" + "\033[0m")
		if self.brain == None:
			brain = Brain(self.steps_num, self.client, self.eval_client, self.base_port, self.eval_port, verbose=self.verbose, DATA_DIR=self.results_dir, transfer=self.transfer, hrl=self.hrl, seed=self.seed, test=self.test)
			if self.load_brain or self.recover:
				brain.load_infos()
		else:
			brain = self.brain
			brain.DATA_DIR = self.results_dir
		# run the pre novelty evaluation on self.trials_eval_pre episodes



		self.trials_eval_pre = 1
		self.trials_training = 5



		if self.trials_eval_pre > 0:
			print("\n\n\nEVALUATION. PRE-NOVELTY")
			# print(f"Num Pre-novelty trials {self.trials_eval_pre}")
			print(f"Num Pre-novelty trials {self.trials_eval_pre}")
			brain.generate_env(env_id=self.env_id, reset_loc="l2", reset_dir="s", render=self.render)
			succ = self.eval(brain, "Pre-novelty", eval_num=self.trials_eval_pre)
			self.write_row_to_results([0, succ], "test")

		# inject novelty and run again evaluation on self.trials_eval_pre episodes
		if self.trials_eval_pre > 0:
			print("\n\n\nEVALUATION. NOVELTY INJECTION")
			print(self.trials_eval_pre)
			# time.sleep(5)
			brain.generate_env(env_id=self.env_id, novelty_list=self.novelty_list, reset_loc="l2", reset_dir="s", render=self.render)
			succ = self.eval(brain, "Pre-training", eval_num=self.trials_eval_pre)
			self.write_row_to_results([1, succ], "test")	

		# train on novelty on self.trials_training episodes 
		if self.trials_training > 0:
			t_start = datetime.now()
			print("\n\n\nTRAINING. Started at: ", t_start)
			print(self.trials_training)
			# time.sleep(5)
			brain.generate_env(env_id=self.env_id, novelty_list=self.novelty_list, reset_loc="l2", reset_dir="s", render=self.render, only_train=True)
			for episode in range(self.trials_training):
				task = self.generate_random_task()
				max_trials = 1
				if self.direct_training:
					max_trials = 1
				
				#Fails here
				done, trial = brain.run_brain(task=task, trial=1, max_trials=max_trials, only_train=True, direct_train=self.direct_training)
				print("\tPost-novelty domain  > Train Success on episode {}: {}\n\n".format(episode, done))
				self.write_row_to_results([episode, done], "train")
			t_end = datetime.now()
			print("\n\n\nTRAINING Stopped at: ", t_end)
			print("oooooooooooooooooooooooooooooooooooooooooo__. Training took {} time .__oooooooooooooooooooooooooooooooooooooooooo".format(t_end-t_start))

		# run the post novelty evaluation on self.trials_eval_post episodes
		if self.trials_eval_post > 0:
			print("\n\n\nEVALUATION. TRAINED ON NOVELTY")
			brain.generate_env(env_id=self.env_id, novelty_list=self.novelty_list, reset_loc="l2", reset_dir="s", render=self.render)
			succ = self.eval(brain, "Post-training", eval_num=self.trials_eval_post)
			self.write_row_to_results([2, succ], "test")

		brain.close_env()
		#if self.trials_eval_pre > 0:
		brain.save_infos()
		print("\n\n\n\n\n\n")
		return brain

	def train_init_policies(self):
		print("\033[1m" + "\n\n\t\t\t\t\t===> RAPIDLEARN EXPERIMENT ON: {} <===\n\n".format(self.novelty) + "\033[0m")
		brain = Brain(self.steps_num, self.client, self.eval_client, self.base_port, self.eval_port, verbose=self.verbose, DATA_DIR=self.results_dir, transfer=self.transfer, seed=self.seed, test=self.test, use_base_policies=False)

		print("\n\n\nTRAINING")
		brain.generate_env(env_id=self.env_id, novelty_list=self.novelty_list, reset_loc="l2", reset_dir="s", render=self.render, only_train=True)
		for episode in range(3):
			task = self.generate_random_task()
			brain.run_brain(task=task, trial=1, max_trials=1, only_train=True, direct_train=True)

		brain.close_env()
		brain.save_infos()
		print("\n\n\n\n\n\n")
		return brain

	def generate_random_task(self, loc=None):
		if loc == None or not(self.fixed_locations):
			goal_location = 2
			while goal_location == 2:
				goal_location = random.randint(0,21)
		else:
			loc -= 1
			loc = (loc % 21) +1
			goal_location = loc
			if goal_location >= 2:
				goal_location += 1
			#if goal_location != 1:
			#	goal_location = 6
		task = "Navigation li l" + str(goal_location)
		return task 

	def eval(self, brain, title, eval_num=100):
		succ = 0
		for episode in range(eval_num):
			try:
				task = self.generate_random_task(episode+1)
				done, trial = brain.run_brain(task=task, only_eval=True, direct_train=self.direct_training)
				print("\t"+title+" domain  > Test Success on episode {}: {}\n\n".format(episode, done))
				if done:
					succ +=1
			except Exception as e:
				print("Exception occured with message {}".format(e)) # error independent of the method performances
				episode -= 1
		return succ/eval_num


class BaselineExperiment(Experiment):
	HEADER_TEST = ['Novel', 'success_rate','tout'] # Novel: 0=pre-novelty_domain, 1=post-novelty_domain
	# Env parameters
	env_config = {'discrete': False, 'stop_prob':0.0, 'max_offset': 4, 'ignore_stops': True, 'change_stop_prob':0}
	eval_config = {'discrete': False, 'stop_prob':0.0, 'end_at_target':True, 'ignore_stops':True, 'eval':True, 'change_stop_prob':0}
	
	def __init__(self, args, env_id='carla-navigate-v0', novelty_list=None, experiment_id='no_id', model=None, eval_only=False, eval_only_pre_post=False):
		if novelty_list == None:
			novelty_list = args['novelty_list']
		if experiment_id == None:
			experiment_id = '_'.join(novelty_list)
		super(BaselineExperiment, self).__init__(args, novelty_list, experiment_id)
		#os.makedirs(self.results_dir, exist_ok=True)

		self.write_row_to_results(self.HEADER_TEST, "test")

		if eval_only_pre_post or self.eval_pre_post:
			self.trials_training = 0
			self.trials_eval_post = 0
		elif eval_only or self.eval_post:
			self.trials_training = 0
			self.trials_eval_pre = 0

		self.reset_loc = None
		self.reset_dir = None
		self.env_id = env_id
		self.novelty_list = novelty_list
		if type(self.novelty_list) == str:
			self.novelty_list = [self.novelty_list]
		self.novelty_env = None
		self.env = None
		self.eval_env = None

		self.name = 'navigate'
		self.folder = self.results_dir
		self.policy_folder = self.folder + "policy/" + self.name
		self.tensorboard_folder = self.folder+"tensorboard/"

		self.client = args['client']
		self.eval_client = args['eval_client']
		self.set_envs(env_id)

		if type(model) is str:
			model = SAC.load(model, env=self.env, tensorboard_log=self.tensorboard_folder, device="cuda") 

		self.novelty = novelty_list
		self.model = model


	def run(self):
		print("\033[1m" + "\n\n\t\t\t\t\t===> RL BASELINE EXPERIMENT ON: {} <===\n\n".format(self.novelty) + "\033[0m")
		if self.model == None:
			print("ERROR: Please enter a valid path to model to load.")
			sys.exit()
		else:
			model = self.model

		# run the pre novelty evaluation on self.trials_eval_pre episodes
		if self.trials_eval_pre > 0:
			print("\n\n\nEVALUATION. PRE-NOVELTY")
			succ, tout = eval_success(self.eval_env, model, loc="l2", direction="s", eval_num=self.trials_eval_pre, fixed_targets=self.fixed_locations)
			print("Pre-novelty Success rate", succ, tout)
			self.write_row_to_results([0, succ, tout], "test")

		# inject novelty and run again evaluation on self.trials_eval_pre episodes
		if self.trials_eval_pre > 0:
			print("\n\n\nEVALUATION. NOVELTY INJECTION")
			self.set_envs(self.env_id, inject=True, novelty_list=self.novelty_list)
			succ, tout = eval_success(self.eval_env, model, loc="l2", direction="s", eval_num=self.trials_eval_pre, fixed_targets=self.fixed_locations)
			print("On novelty, pre-training Success rate", succ, tout)
			self.write_row_to_results([1, succ, tout], "test")	

		# inject novelty and run online evaluation
		if self.trials_training > 0:
			t_start = datetime.now()
			print("\n\n\nTRAINING. Started at: ", t_start)
			self.set_envs(self.env_id, inject=True, novelty_list=self.novelty_list)
			model = self.learn_policy()
			t_end = datetime.now()
			print("\n\n\nTRAINING Stopped at: ", t_end)
			print("oooooooooooooooooooooooooooooooooooooooooo__. Training took {} time .__oooooooooooooooooooooooooooooooooooooooooo".format(t_end-t_start))

		# run the post novelty evaluation on self.trials_eval_post episodes
		if self.trials_eval_post > 0:
			print("\n\n\nEVALUATION. TRAINED ON NOVELTY")
			self.set_envs(self.env_id, inject=True, novelty_list=self.novelty_list)
			succ, tout = eval_success(self.eval_env, model, loc="l2", direction="s", eval_num=self.trials_eval_post, fixed_targets=self.fixed_locations)
			print("After training Success rate", succ, tout)
			self.write_row_to_results([2, succ, tout], "test")

		self.close_envs()
		print("\n\n\n\n\n\n")
		return model

	def close_envs(self):
		if self.env != None:
			self.env.close()
			self.env = None 
			print("Closing training env.")
		if self.eval_env != None:
			self.eval_env.close()
			self.eval_env = None 
			print("Closing eval env.")

	def set_envs(self, env_id, inject=False, novelty_list=None):
		if self.env == None or novelty_list != self.novelty_env:
			if self.env != None or self.eval_env != None:
				self.close_envs()

			runtime_settings = {'address': ["localhost", self.base_port], 'no_rendering': True, 'client': self.client, 'reload_world': False}
			eval_runtime_settings = {'address': ["localhost", self.base_port+self.eval_port],'no_rendering': not(self.render), 'client': self.eval_client, 'reload_world': False}

			print("\nCreating training env.")
			env = gym.make(env_id, params=self.env_config, runtime_settings=runtime_settings)
			print("Creating eval env.")
			eval_env = gym.make(env_id, params=self.eval_config, runtime_settings=eval_runtime_settings)

			# Injecting the novelties in the environment
			if (self.novelty_list is not None) and inject:
				if self.reset_loc == None:
					print("Training from random initial state: reset_loc={}, reset_dir={}".format(self.reset_loc, self.reset_dir))
					env = NoveltyWrapper(env)
					eval_env = NoveltyWrapper(eval_env)
				else:
					print("Training from fixed initial state: reset_loc={}, reset_dir={}".format(self.reset_loc, self.reset_dir))
					env = NoveltyWrapper(env, loc=self.reset_loc, direction=self.reset_dir)
					eval_env = NoveltyWrapper(eval_env, loc=self.reset_loc, direction=self.reset_dir)
				if type(self.novelty_list) == str:
					self.novelty_list = [self.novelty_list]
				novelties = []
				novelties_eval = []
				for novelty in self.novelty_list:
					print("\033[1m" + "\n\n\t===> Injecting Novelty: {} <===\n\n".format(novelty) + "\033[0m")
					novel_wrapper = novelties_info[novelty]["wrapper"]
					novel_params = novelties_info[novelty]["params"]
					if novel_params is None:
						novel_params = {}
					novelties.append(novel_wrapper(env=env, **novel_params))
					novelties_eval.append(novel_wrapper(env=eval_env, **novel_params))
				env.set_novelties(novelties)
				eval_env.set_novelties(novelties_eval)

			env.seed(self.seed)
			eval_env.seed(self.seed)

			self.env = env  
			self.eval_env = eval_env
			self.novelty_env = novelty_list

	def learn_policy(self):

		if self.model == None:
			print("ERROR: Please enter a valid path to model to load.")
			sys.exit()
		else:
			model = self.model

		model.set_env(self.env)
		parameters = model.get_parameters()
		model.set_parameters(parameters)

		self.training_steps = self.steps_num
		self.eval_freq = int(self.steps_num/5)

		policy_kwargs = dict(net_arch=dict(pi=[512, 512, 512], qf=[512, 512, 512]))

		if self.test:
			model= train(
				self.env, 
				eval_env=self.eval_env, 
				model=model, 
				policy_kwargs=policy_kwargs,
				reward_threshold=810,
				save_freq = 200, 
				total_timesteps = 300,
				best_model_save_path=self.policy_folder,
				eval_freq=200, 
				n_eval_episodes=1,
				save_path=self.folder)
		else:
			model= train(
				self.env, 
				eval_env=self.eval_env, 
				model=model, 
				policy_kwargs=policy_kwargs,
				reward_threshold=810,
				save_freq = self.eval_freq, 
				total_timesteps = self.training_steps,
				best_model_save_path=self.policy_folder,
				eval_freq=self.eval_freq, 
				n_eval_episodes=20,
				save_path=self.folder)

		self.model = model
		novelty_pattern_name = '_'.join(sorted(self.novelty_env))
		self.model.save(self.results_dir + novelty_pattern_name)
		print('Saving in ', self.results_dir + novelty_pattern_name)
		return model

def to_datestring(unixtime: int, format='%Y-%m-%d_%H:%M:%S'):
	return datetime.utcfromtimestamp(unixtime).strftime(format)


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--experiment", default="experiment", help="Type of experiment: can be rapid (script), baseline(script), vanilla_rapid(script), vanilla_baseline(script), RL(per novelty) or nothing=rapid(per novelty).")
	ap.add_argument("-N", "--novelty_list", nargs='+', default=None, help="List of novelty (-N n1 n2...) to inject including: #black_ice #rain #mist #deflated_tire #obstacle #traffic", type=str)
	ap.add_argument("-log", "--log_storage_folder", default='data/', help="Path to log storage folder", type=str)
	ap.add_argument("-te", "--trials_eval", default=100, help="Number of episode to evaluate the agent pre and post novelty performances", type=int)
	ap.add_argument("-tt", "--trials_training", default=4, help="Number of episodes of novelty accomodation", type=int)
	ap.add_argument("-steps", "--steps", default=2_000_000, help="Number of steps to train on each novelty", type=int)
	ap.add_argument("-s", "--seed", default=0, help="Experiment seed", type=int)
	ap.add_argument("-p", "--carla_base_port", default=2000, help="Carla server port", type=int)
	ap.add_argument("-pe", "--carla_eval_port", default=2, help="Carla eval server port (base port + pe int)", type=int)
	ap.add_argument("-m", "--model", help="Path to file model to load for SAC RL baseline and folder containing basic operators for rapid.", default='models/navigate', type=str)
	ap.add_argument("-V", "--verbose", action="store_true", default=False, help="Boolean: verbose.")
	ap.add_argument("-T", "--transfer", action="store_true", default=False, help="Boolean: transfers policy from a source policy selected by the function select_closest_pattern in domain_synapses.py.")
	ap.add_argument("-R", "--render", action="store_true", default=False, help="Boolean: renders the environment.")
	ap.add_argument("-half", "--half", action="store_true", default=False, help="Boolean: cuts the experiments in two parts.")
	ap.add_argument("-F", "--fixed_locations", action="store_true", default=False, help="Boolean: sets the eval locations to the fixed set of all 20 high-level locations.")
	ap.add_argument("-quick_training", "--test", action="store_true", default=False, help="Boolean: sets low number of training episodes")
	ap.add_argument("-direct_training", "--direct_training", action="store_true", default=False, help="Boolean: sets operator failure to True with pre-defined failed operator and failure states: cf brain file direct_training dict.")
	ap.add_argument("-eval", "--eval", action="store_true", default=False, help="Boolean: sets the experiment for eval only after novelty injection.")
	ap.add_argument("-eval_pre_post", "--eval_pre_post", action="store_true", default=False, help="Boolean: sets the experiment for eval only before and after novelty injection.")
	ap.add_argument("-pre_training", "--pre_training", action="store_true", default=False, help="Boolean: regenerates initial policies for goForward, turnLeft and turnRight.")
	ap.add_argument("-recover", "--recover", action="store_true", default=False, help="Boolean: recover brain from -log <experiment_path>.")
	ap.add_argument("-load", "--load", action="store_true", default=False, help="Boolean load brain from -log <experiment_path>.")
	ap.add_argument("-hrl", "--hrl", action="store_true", default=False, help="Boolean: use of hierarchical RL when facing local novelties.")


	args = vars(ap.parse_args())

	args['client'] = conf_carla(args['carla_base_port'])
	args['eval_client'] = conf_carla(args['carla_base_port']+args['carla_eval_port'])

	set_random_seed(args['seed'], using_cuda=True)
	random.seed(args['seed'])

	if args['novelty_list'] == None:
		args['novelty_list'] = []

	checkpoint = args['novelty_list']
	if checkpoint == []:
		reached_checkpoint = True  
	else: 
		reached_checkpoint = False
		model = args['model']

	if args['load']:
		ex_id = ''
	elif not(args['recover']):
		ex_id = args['experiment'] + f"{to_datestring(time.time())}"#self.hashid
		for exec_id in executors_id_list:
			if args['model'] == 'models/navigate':
				path_to_model = './gym_carla_novelty/policies/' + exec_id
			else:
				if args['model'].endswith('/'):
					path_to_model = args['model'] + exec_id
				else:
					path_to_model = args['model'] + '/' + exec_id
			if args['experiment'] != 'baseline':
				print("Loading basic executor: ", path_to_model)
				executor = Executor(id=exec_id, policy=path_to_model, Beta=beta_indicator, basic=True)
				ex_dict = {exec_id:executor}
				executors.update(ex_dict)
	else:
		save_log = args['log_storage_folder']
		save_model = args['model']
		save_client = args['client']
		save_eval_client = args['eval_client']
		save_base_port = args['carla_base_port']
		save_eval_port = args['carla_eval_port']
		with open(args['log_storage_folder']+'/args.txt', 'r') as f:
			args = json.load(f)
		args['recover'] = True
		args['log_storage_folder'] = save_log
		args['model'] = save_model
		args['client'] = save_client
		args['eval_client'] = save_eval_client
		args['carla_base_port'] = save_base_port
		args['carla_eval_port'] = save_eval_port
		print('Loagind arguments from checkpoint. Args are: \n', args)
		ex_id = ''
		if checkpoint == []:
			checkpoint = args['checkpoint']
			reached_checkpoint = False
			print('\n==> RECOVERING FROM {} <=='.format(checkpoint))
			if args['experiment'] == 'baseline':
				if save_model == 'models/navigate' and args['previous'] != []: # If it consists of the default value, overwrites model using the previous checkpoint model
					name_novelty = ("_").join([novelty.split("_")[-1] for novelty in args['previous']])
					policy_file = '_'.join(sorted(args['previous']))
					model = ("/").join([args['log_storage_folder'], name_novelty, policy_file])
				print('\n==> WITH INITIAL POLICY {} <=='.format(model))

	if  args['experiment'] == 'rapid':

		brain = None
		if args['pre_training']:
			init = RapidExperiment(args, brain=brain, experiment_id=ex_id+"/init")
			brain = init.train_init_policies()

		# Novelty 'deflated_tire'
		if reached_checkpoint or checkpoint == ['deflated_tire']:
			args['checkpoint'], args['previous'] = ['deflated_tire'], []
			experiment1 = RapidExperiment(args, novelty_list=['deflated_tire'], brain=brain, experiment_id=ex_id+"/tire")
			brain = experiment1.run()
			reached_checkpoint = True
			args['previous'] = []

		# Novelty 'black_ice'
		if reached_checkpoint or checkpoint == ['black_ice']:
			args['checkpoint'], args['previous'] = ['black_ice'], ['deflated_tire']
			experiment2 = RapidExperiment(args, novelty_list=['black_ice'], brain=brain, experiment_id=ex_id+"/ice")
			brain = experiment2.run()
			eval = RapidExperiment(args, novelty_list=['deflated_tire'], brain=brain, experiment_id=ex_id+"/ice"+"/tire_eval", eval_only=True)
			eval.run()
			reached_checkpoint = True

		# Novelty 'deflated_tire','rain'
		if reached_checkpoint or checkpoint == ['deflated_tire','rain']:
			args['checkpoint'], args['previous'] = ['deflated_tire','rain'], ['black_ice']
			experiment3 = RapidExperiment(args, novelty_list=['deflated_tire','rain'], brain=brain, experiment_id=ex_id+"/tire_rain")
			brain = experiment3.run()
			eval = RapidExperiment(args, novelty_list=['deflated_tire'], brain=brain, experiment_id=ex_id+"/tire_rain"+"/tire_eval", eval_only=True)
			eval.run()
			eval = RapidExperiment(args, novelty_list=['black_ice'], brain=brain, experiment_id=ex_id+"/tire_rain"+"/ice_eval", eval_only=True)
			eval.run()
			reached_checkpoint = True

		if not(args['half']):
			# Novelty 'obstacle'
			if reached_checkpoint or checkpoint == ['obstacle']:
				args['checkpoint'], args['previous'] = ['obstacle'], ['deflated_tire','rain']
				experiment4 = RapidExperiment(args, novelty_list=['obstacle'], brain=brain, experiment_id=ex_id+"/obstacle")
				brain = experiment4.run()
				eval = RapidExperiment(args, novelty_list=['deflated_tire'], brain=brain, experiment_id=ex_id+"/obstacle"+"/tire_eval", eval_only=True)
				eval.run()
				eval = RapidExperiment(args, novelty_list=['black_ice'], brain=brain, experiment_id=ex_id+"/obstacle"+"/ice_eval", eval_only=True)
				eval.run()
				eval = RapidExperiment(args, novelty_list=['deflated_tire','rain'], brain=brain, experiment_id=ex_id+"/obstacle"+"/tire_rain_eval", eval_only=True)
				eval.run()
				reached_checkpoint = True

			# Novelty 'obstacle','black_ice'
			if reached_checkpoint or checkpoint == ['obstacle','black_ice']:
				args['checkpoint'], args['previous'] = ['obstacle','black_ice'], ['obstacle']
				experiment6 = RapidExperiment(args, novelty_list=['obstacle','black_ice'], brain=brain, experiment_id=ex_id+"/obstacle_ice")
				brain = experiment6.run()	
				eval = RapidExperiment(args, novelty_list=['deflated_tire'], brain=brain, experiment_id=ex_id+"/obstacle_ice"+"/tire_eval", eval_only=True)
				eval.run()
				eval = RapidExperiment(args, novelty_list=['black_ice'], brain=brain, experiment_id=ex_id+"/obstacle_ice"+"/ice_eval", eval_only=True)
				eval.run()
				eval = RapidExperiment(args, novelty_list=['deflated_tire','rain'], brain=brain, experiment_id=ex_id+"/obstacle_ice"+"/tire_rain_eval", eval_only=True)
				eval.run()
				eval = RapidExperiment(args, novelty_list=['obstacle'], brain=brain, experiment_id=ex_id+"/obstacle_ice"+"/obstacle_eval", eval_only=True)
				eval.run()
				reached_checkpoint = True

			# Novelty 'obstacle','mist'
			if reached_checkpoint or checkpoint == ['obstacle','mist']:
				args['checkpoint'], args['previous'] = ['obstacle','mist'], ['obstacle','black_ice']
				experiment5 = RapidExperiment(args, novelty_list=['obstacle','mist'], brain=brain, experiment_id=ex_id+"/obstacle_mist")
				brain = experiment5.run()
				eval = RapidExperiment(args, novelty_list=['deflated_tire'], brain=brain, experiment_id=ex_id+"/obstacle_mist"+"/tire_eval", eval_only=True)
				eval.run()
				eval = RapidExperiment(args, novelty_list=['black_ice'], brain=brain, experiment_id=ex_id+"/obstacle_mist"+"/ice_eval", eval_only=True)
				eval.run()
				eval = RapidExperiment(args, novelty_list=['deflated_tire','rain'], brain=brain, experiment_id=ex_id+"/obstacle_mist"+"/tire_rain_eval", eval_only=True)
				eval.run()
				eval = RapidExperiment(args, novelty_list=['obstacle'], brain=brain, experiment_id=ex_id+"/obstacle_mist"+"/obstacle_eval", eval_only=True)
				eval.run()
				eval = RapidExperiment(args, novelty_list=['obstacle','black_ice'], brain=brain, experiment_id=ex_id+"/obstacle_mist"+"/obstacle_ice_eval", eval_only=True)
				eval.run()
				reached_checkpoint = True

			# Novelty 'traffic'
			if reached_checkpoint or checkpoint == ['traffic']:
				experiment11 = RapidExperiment(args, novelty_list=['traffic'], brain=brain, experiment_id=ex_id+"/traffic")
				brain = experiment11.run()	
				#eval = RapidExperiment(args, novelty_list=['deflated_tire'], brain=brain, experiment_id=ex_id+"/traffic"+"/tire_eval", eval_only=True)
				#eval.run()
				#eval = RapidExperiment(args, novelty_list=['black_ice'], brain=brain, experiment_id=ex_id+"/traffic"+"/ice_eval", eval_only=True)
				#eval.run()
				#eval = RapidExperiment(args, novelty_list=['deflated_tire','rain'], brain=brain, experiment_id=ex_id+"/traffic"+"/tire_rain_eval", eval_only=True)
				#eval.run()
				#eval = RapidExperiment(args, novelty_list=['obstacle'], brain=brain, experiment_id=ex_id+"/traffic"+"/obstacle_eval", eval_only=True)
				#eval.run()
				#eval = RapidExperiment(args, novelty_list=['obstacle','mist'], brain=brain, experiment_id=ex_id+"/traffic"+"/obstacle_mist_eval", eval_only=True)
				#eval.run()
				#eval = RapidExperiment(args, novelty_list=['obstacle','mist','rain'], brain=brain, experiment_id=ex_id+"/traffic"+"/obstacle_mist_rain_eval", eval_only=True)
				#eval.run()
				reached_checkpoint = True
				
			# Novelty 'traffic','rain'
			if reached_checkpoint or checkpoint == ['traffic','rain']:
				experiment12 = RapidExperiment(args, novelty_list=['traffic','rain'], brain=brain, experiment_id=ex_id+"/traffic_rain")
				brain = experiment12.run()
				#eval = RapidExperiment(args, novelty_list=['deflated_tire'], brain=brain, experiment_id=ex_id+"/traffic_rain"+"/tire_eval", eval_only=True)
				#eval.run()
				#eval = RapidExperiment(args, novelty_list=['black_ice'], brain=brain, experiment_id=ex_id+"/traffic_rain"+"/ice_eval", eval_only=True)
				#eval.run()
				#eval = RapidExperiment(args, novelty_list=['deflated_tire','rain'], brain=brain, experiment_id=ex_id+"/traffic_rain"+"/tire_rain_eval", eval_only=True)
				#eval.run()
				#eval = RapidExperiment(args, novelty_list=['obstacle'], brain=brain, experiment_id=ex_id+"/traffic_rain"+"/obstacle_eval", eval_only=True)
				#eval.run()
				#eval = RapidExperiment(args, novelty_list=['obstacle','mist'], brain=brain, experiment_id=ex_id+"/traffic_rain"+"/obstacle_mist_eval", eval_only=True)
				#eval.run()
				#eval = RapidExperiment(args, novelty_list=['obstacle','mist','rain'], brain=brain, experiment_id=ex_id+"/traffic_rain"+"/obstacle_mist_rain_eval", eval_only=True)
				#eval.run()
				#eval = RapidExperiment(args, novelty_list=['traffic'], brain=brain, experiment_id=ex_id+"/traffic_rain"+"/traffic", eval_only=True)
				#eval.run()
				
				experiment10 = RapidExperiment(args, novelty_list=['obstacle','rain'], brain=brain, experiment_id=ex_id+"/traffic_rain"+"/obstacle_rain_eval", eval_only=True)
				brain = experiment10.run()
				experiment_noise1 = RapidExperiment(args, novelty_list=['obstacle_jitter'], brain=brain, experiment_id=ex_id+"/traffic_rain"+"/obstacle_jitter", eval_only=True)
				brain = experiment_noise1.run()
				experiment_noise2 = RapidExperiment(args, novelty_list=['rain_jitter'], brain=brain, experiment_id=ex_id+"/traffic_rain"+"/rain_jitter", eval_only=True)
				brain = experiment_noise2.run()
				experiment_noise3 = RapidExperiment(args, novelty_list=['black_ice_jitter'], brain=brain, experiment_id=ex_id+"/traffic_rain"+"/black_ice_jitter", eval_only=True)
				brain = experiment_noise3.run()
				experiment7 = RapidExperiment(args, novelty_list=['obstacle','mist','black_ice'], brain=brain, experiment_id=ex_id+"/traffic_rain"+"/obstacle_mist_ice_eval", eval_only=True)
				brain = experiment7.run()

				reached_checkpoint = True

	elif args['experiment'] == 'baseline':

		# Novelty 'deflated_tire'
		if reached_checkpoint or checkpoint == ['deflated_tire']:
			args['checkpoint'], args['previous'] = ['deflated_tire'], []
			experiment1 = BaselineExperiment(args, novelty_list=['deflated_tire'], model=args['model'], experiment_id=ex_id+"/tire")
			model = experiment1.run()
			reached_checkpoint = True

		# Novelty 'black_ice'
		if reached_checkpoint or checkpoint == ['black_ice']:
			args['checkpoint'], args['previous'] = ['black_ice'], ['deflated_tire']
			experiment2 = BaselineExperiment(args, novelty_list=['black_ice'], model=model, experiment_id=ex_id+"/ice")
			model = experiment2.run()
			eval = BaselineExperiment(args, novelty_list=['deflated_tire'], model=model, experiment_id=ex_id+"/ice"+"/tire_eval", eval_only=True)
			eval.run()
			reached_checkpoint = True

		# Novelty 'deflated_tire','rain'
		if reached_checkpoint or checkpoint == ['deflated_tire','rain']:
			args['checkpoint'], args['previous'] = ['deflated_tire','rain'], ['black_ice']
			experiment3 = BaselineExperiment(args, novelty_list=['deflated_tire','rain'], model=model, experiment_id=ex_id+"/tire_rain")
			model = experiment3.run()
			eval = BaselineExperiment(args, novelty_list=['deflated_tire'], model=model, experiment_id=ex_id+"/tire_rain"+"/tire_eval", eval_only=True)
			eval.run()
			eval = BaselineExperiment(args, novelty_list=['black_ice'], model=model, experiment_id=ex_id+"/tire_rain"+"/ice_eval", eval_only=True)
			eval.run()
			reached_checkpoint = True

		if not(args['half']):
			# Novelty 'obstacle'
			if reached_checkpoint or checkpoint == ['obstacle']:
				args['checkpoint'], args['previous'] = ['obstacle'], ['deflated_tire','rain']
				experiment4 = BaselineExperiment(args, novelty_list=['obstacle'], model=model, experiment_id=ex_id+"/obstacle")
				model = experiment4.run()
				eval = BaselineExperiment(args, novelty_list=['deflated_tire'], model=model, experiment_id=ex_id+"/obstacle"+"/tire_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['black_ice'], model=model, experiment_id=ex_id+"/obstacle"+"/ice_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['deflated_tire','rain'], model=model, experiment_id=ex_id+"/obstacle"+"/tire_rain_eval", eval_only=True)
				eval.run()
				reached_checkpoint = True

			# Novelty 'obstacle','black_ice'
			if reached_checkpoint or checkpoint == ['obstacle','black_ice']:
				args['checkpoint'], args['previous'] = ['obstacle','black_ice'], ['obstacle']
				experiment5 = BaselineExperiment(args, novelty_list=['obstacle','black_ice'], model=model, experiment_id=ex_id+"/obstacle_ice")
				model = experiment5.run()
				eval = BaselineExperiment(args, novelty_list=['deflated_tire'], model=model, experiment_id=ex_id+"/obstacle_ice"+"/tire_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['black_ice'], model=model, experiment_id=ex_id+"/obstacle_ice"+"/ice_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['deflated_tire','rain'], model=model, experiment_id=ex_id+"/obstacle_ice"+"/tire_rain_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['obstacle'], model=model, experiment_id=ex_id+"/obstacle_ice"+"/obstacle_eval", eval_only=True)
				eval.run()
				reached_checkpoint = True

			# Novelty 'obstacle','mist'
			if reached_checkpoint or checkpoint == ['obstacle','mist']:
				args['checkpoint'], args['previous'] = ['obstacle','mist'], ['obstacle','black_ice']
				experiment6 = BaselineExperiment(args, novelty_list=['obstacle','mist'], model=model, experiment_id=ex_id+"/obstacle_mist")
				model = experiment6.run()
				eval = BaselineExperiment(args, novelty_list=['deflated_tire'], model=model, experiment_id=ex_id+"/obstacle_mist"+"/tire_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['black_ice'], model=model, experiment_id=ex_id+"/obstacle_mist"+"/ice_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['deflated_tire','rain'], model=model, experiment_id=ex_id+"/obstacle_mist"+"/tire_rain_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['obstacle'], model=model, experiment_id=ex_id+"/obstacle_mist"+"/obstacle_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['obstacle','black_ice'], model=model, experiment_id=ex_id+"/obstacle_mist"+"/obstacle_ice_eval", eval_only=True)
				eval.run()
				reached_checkpoint = True

			# Novelty 'traffic'
			if reached_checkpoint or checkpoint == ['traffic']:
				experiment11 = BaselineExperiment(args, novelty_list=['traffic'], model=model, experiment_id=ex_id+"/traffic")
				model = experiment11.run()
				eval = BaselineExperiment(args, novelty_list=['deflated_tire'], model=model, experiment_id=ex_id+"/traffic"+"/tire_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['black_ice'], model=model, experiment_id=ex_id+"/traffic"+"/ice_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['deflated_tire','rain'], model=model, experiment_id=ex_id+"/traffic"+"/tire_rain_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['obstacle'], model=model, experiment_id=ex_id+"/traffic"+"/obstacle_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['obstacle','mist'], model=model, experiment_id=ex_id+"/traffic"+"/obstacle_mist_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['obstacle','mist','rain'], model=model, experiment_id=ex_id+"/traffic"+"/obstacle_mist_rain_eval", eval_only=True)
				eval.run()
				reached_checkpoint = True

			# Novelty 'traffic','rain'
			if reached_checkpoint or checkpoint == ['traffic','rain']:
				experiment12 = BaselineExperiment(args, novelty_list=['traffic','rain'], model=model, experiment_id=ex_id+"/traffic_rain")
				model = experiment12.run()
				eval = BaselineExperiment(args, novelty_list=['deflated_tire'], model=model, experiment_id=ex_id+"/traffic_rain"+"/tire_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['black_ice'], model=model, experiment_id=ex_id+"/traffic_rain"+"/ice_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['deflated_tire','rain'], model=model, experiment_id=ex_id+"/traffic_rain"+"/tire_rain_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['obstacle'], model=model, experiment_id=ex_id+"/traffic_rain"+"/obstacle_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['obstacle','mist'], model=model, experiment_id=ex_id+"/traffic_rain"+"/obstacle_mist_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['obstacle','mist','rain'], model=model, experiment_id=ex_id+"/traffic_rain"+"/obstacle_mist_rain_eval", eval_only=True)
				eval.run()
				eval = BaselineExperiment(args, novelty_list=['traffic'], model=model, experiment_id=ex_id+"/traffic_rain"+"/traffic_eval", eval_only=True)
				eval.run()

				experiment10 = BaselineExperiment(args, novelty_list=['obstacle','rain'], model=model, experiment_id=ex_id+"/traffic_rain"+"/obstacle_rain_eval", eval_only=True)
				model = experiment10.run()
				experiment_noise1 = BaselineExperiment(args, novelty_list=['obstacle_jitter'], model=model, experiment_id=ex_id+"/traffic_rain"+"/obstacle_jitter", eval_only=True)
				model = experiment_noise1.run()
				experiment_noise2 = BaselineExperiment(args, novelty_list=['rain_jitter'], model=model, experiment_id=ex_id+"/traffic_rain"+"/rain_jitter", eval_only=True)
				model = experiment_noise2.run()
				experiment_noise3 = BaselineExperiment(args, novelty_list=['black_ice_jitter'], model=model, experiment_id=ex_id+"/traffic_rain"+"/black_ice_jitter", eval_only=True)
				model = experiment_noise3.run()
				experiment7 = BaselineExperiment(args, novelty_list=['obstacle','mist','black_ice'], model=model, experiment_id=ex_id+"/traffic_rain"+"/obstacle_mist_ice_eval", eval_only=True)
				model = experiment7.run()
				reached_checkpoint = True



	elif args['experiment'] == 'traffic_baseline':

		experiment11 = BaselineExperiment(args, novelty_list=['traffic'], model=args['model'], experiment_id=ex_id+"/traffic")
		model = experiment11.run()
		experiment12 = BaselineExperiment(args, novelty_list=['traffic','rain'], model=model, experiment_id=ex_id+"/traffic_rain")
		model = experiment12.run()
		experiment2 = BaselineExperiment(args, novelty_list=['black_ice'], model=model, experiment_id=ex_id+"/ice")
		model = experiment2.run()

	elif args['experiment'] == 'traffic_rapid':

		experiment11 = RapidExperiment(args, novelty_list=['traffic'], experiment_id=ex_id+"/traffic")
		brain = experiment11.run()	
		experiment12 = RapidExperiment(args, novelty_list=['traffic','rain'], brain=brain, experiment_id=ex_id+"/traffic_rain")
		brain = experiment12.run()
		experiment2 = RapidExperiment(args, novelty_list=['black_ice'], brain=brain, experiment_id=ex_id+"/ice")
		brain = experiment2.run()

	elif args['experiment'] == 'vanilla_rapid':

		RapidExperiment(args, novelty_list=['deflated_tire'], experiment_id=ex_id+"/tire", eval_only=True).run()
		RapidExperiment(args, novelty_list=['black_ice'], experiment_id=ex_id+"/ice_", eval_only=True).run()
		RapidExperiment(args, novelty_list=['deflated_tire','rain'], experiment_id=ex_id+"/tire_rain", eval_only=True).run()
		RapidExperiment(args, novelty_list=['obstacle'], experiment_id=ex_id+"/obstacle", eval_only=True).run()
		RapidExperiment(args, novelty_list=['obstacle','black_ice'], experiment_id=ex_id+"/obstacle_mist", eval_only=True).run()
		RapidExperiment(args, novelty_list=['obstacle','mist'], experiment_id=ex_id+"/obstacle_mist", eval_only=True).run()
		#RapidExperiment(args, novelty_list=['obstacle','black_ice'], experiment_id=ex_id+"/obstacle_ice", eval_only=True).run()
		#RapidExperiment(args, novelty_list=['traffic'], experiment_id=ex_id+"/traffic", eval_only=True).run()
		#RapidExperiment(args, novelty_list=['traffic','rain'], experiment_id=ex_id+"/traffic_rain", eval_only=True).run()

	elif args['experiment'] == 'vanilla_baseline':

		BaselineExperiment(args, novelty_list=['deflated_tire'], model=args['model'], experiment_id=ex_id+"/tire", eval_only=True).run()
		BaselineExperiment(args, novelty_list=['black_ice'], model=args['model'], experiment_id=ex_id+"/ice", eval_only=True).run()
		BaselineExperiment(args, novelty_list=['deflated_tire','rain'], model=args['model'], experiment_id=ex_id+"/tire_rain", eval_only=True).run()
		BaselineExperiment(args, novelty_list=['obstacle'], model=args['model'], experiment_id=ex_id+"/obstacle", eval_only=True).run()
		BaselineExperiment(args, novelty_list=['obstacle','black_ice'], model=args['model'], experiment_id=ex_id+"/obstacle_mist", eval_only=True).run()
		BaselineExperiment(args, novelty_list=['obstacle','mist'], model=args['model'], experiment_id=ex_id+"/obstacle_mist", eval_only=True).run()
		#BaselineExperiment(args, novelty_list=['obstacle','black_ice'], model=args['model'], experiment_id=ex_id+"/obstacle_ice", eval_only=True).run()
		#BaselineExperiment(args, novelty_list=['traffic'], model=args['model'], experiment_id=ex_id+"/traffic", eval_only=True).run()
		#BaselineExperiment(args, novelty_list=['traffic','rain'], model=args['model'], experiment_id=ex_id+"/traffic_rain", eval_only=True).run()

	elif args['experiment'] == 'RL':

		print('RL EXPERIMENT')
		experiment = BaselineExperiment(args, model=args['model'], eval_only=args['eval'], experiment_id=ex_id)
		model = experiment.run()

	else:
		experiment = RapidExperiment(args, experiment_id=ex_id, eval_only=args['eval'], eval_only_pre_post=args['eval_pre_post'])
		brain = experiment.run()

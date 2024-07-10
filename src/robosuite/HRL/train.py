import os
import sys
import numpy as np
import gym_carla_novelty
from gym_carla_novelty.operator_learners.training_wrapper import TrainingWrapper
from gym_carla_novelty.operator_learners.sb3_eval import FixedSeedEvalCallback

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement

save_path = "./"
save_freq = 100000
total_timesteps = 1000000
policy_kwargs = dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]))

def train(env, eval_env=None, model=None, reward_threshold=800,
          save_path=save_path, run_id="", save_freq=save_freq, best_model_save_path=None,
          total_timesteps=total_timesteps, policy_kwargs=policy_kwargs,
          eval_freq=50_000, n_eval_episodes=10, eval_seed=100):
 
    # setting up logging
    if run_id == "":
        if len(sys.argv) > 1:
            run_id = f"-{sys.argv[1]}"
        log_dir = f"{save_path}logs/{env.spec.id}"
        model_dir = f"{save_path}models/{env.spec.id}"

    log_dir = f"{save_path}logs/"
    model_dir = f"{save_path}models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # wrappers
    env = TrainingWrapper(env)
    env = Monitor(env, log_dir)

    # evaluation
    callbacks = []
    if eval_env is not None:
        os.makedirs(log_dir+"-eval", exist_ok=True)
        eval_env = Monitor(eval_env, log_dir+"-eval")
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=2, verbose=1)
        eval_callback = FixedSeedEvalCallback(eval_env,
                                              callback_on_new_best=callback_on_best,
                                              callback_after_eval=stop_train_callback,
                                              n_eval_episodes=n_eval_episodes,
                                              eval_freq=eval_freq,
                                              deterministic=True,
                                              render=False,
                                              best_model_save_path=best_model_save_path,
                                              seed=eval_seed)
        callbacks.append(eval_callback)

    # start learning
    if model == None:
        model = SAC(MlpPolicy, env, verbose=1, gamma=0.95, learning_rate=0.0003, policy_kwargs=policy_kwargs, tensorboard_log=log_dir+"-tensorboard")
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=model_dir, name_prefix=run_id)
    callbacks.append(checkpoint_callback)
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks), log_interval=10)

    

    
    #list_ep_lengths = eval_callback.evaluations_length
    #mean_ep_length_list, std_ep_length_list = [np.mean(episode_lengths) for episode_lengths in list_ep_lengths], [np.std(episode_lengths) for episode_lengths in list_ep_lengths]

    #longest_episode_length = max(mean_ep_length_list)
    #index = mean_ep_length_list.index(longest_episode_length)
    #std_of_longest_episode_length = std_ep_length_list.index(index)

    return model#, eval_callback.best_mean_reward, longest_episode_length, std_of_longest_episode_length

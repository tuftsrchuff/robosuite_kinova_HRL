import numpy as np
import robosuite as suite
from robosuite.utils.mjmod import DynamicsModder
from robosuite.wrappers.behavior_cloning.hanoi_reach_pick import ReachPickWrapper
from robosuite.wrappers.behavior_cloning.hanoi_reach_drop import ReachDropWrapper
from robosuite.wrappers.split_wrappers import PickWrapper
import os
from stable_baselines3.common.env_checker import check_env
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor




def train_reach_pick(env, eval_env):
    #Wrap environment in the wrapper, try to train
    env = ReachPickWrapper(env)
    check_env(env)
    env = Monitor(env, filename=None, allow_early_resets=True)

    # Define the model
    model = SAC(
        'MlpPolicy',
        env,
        buffer_size=int(1e6),
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log='./logs/'
    )

    eval_env = ReachPickWrapper(eval_env)
    eval_env = Monitor(eval_env, filename=None, allow_early_resets=True)

    # Define the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_on_new_best=None,
        verbose=1
    )

    # Train the model
    model.learn(
        total_timesteps=int(1e6),
        callback=eval_callback
    )

    # Save the model
    model.save(os.path.join('./models/reachpick_sac'))

def train_pick_drop(env):
    #Wrap environment in the wrapper, try to train
    env = PickWrapper(env)
    check_env(env)
    env = Monitor(env, filename=None, allow_early_resets=True)

    # Define the model
    model = SAC(
        'MlpPolicy',
        env,
        buffer_size=int(1e6),
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log='./logs/'
    )

    eval_env = PickWrapper(eval_env)
    eval_env = Monitor(eval_env, filename=None, allow_early_resets=True)

    # Define the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_on_new_best=None,
        verbose=1
    )

    # Train the model
    model.learn(
        total_timesteps=int(1e6),
        callback=eval_callback
    )

    # Save the model
    model.save(os.path.join('./models/pick_sac'))

def train_reach_drop(env):
    #Wrap environment in the wrapper, try to train
    env = ReachDropWrapper(env)
    check_env(env)
    env = Monitor(env, filename=None, allow_early_resets=True)

    # Define the model
    model = SAC(
        'MlpPolicy',
        env,
        buffer_size=int(1e6),
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log='./logs/'
    )

    eval_env = ReachDropWrapper(eval_env)
    eval_env = Monitor(eval_env, filename=None, allow_early_resets=True)

    # Define the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_on_new_best=None,
        verbose=1
    )

    # Train the model
    model.learn(
        total_timesteps=int(1e6),
        callback=eval_callback
    )

    # Save the model
    model.save(os.path.join('./models/reachdrop_sac'))



if __name__ == "__main__":
    #Learn decomposed tasks for operators
    print("What policy would you like to train?")
    print("\t1. Reach-pick")
    print("\t2. Pick-drop")
    print("\t3. Reach-drop")
    policy_train_selection = int(input())

    # create environment instance
    env = suite.make(
        env_name="Hanoi",
        robots="Kinova3", 
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera="agentview",
    )

    eval_env = suite.make(
        env_name="Hanoi",
        robots="Kinova3", 
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera="agentview",
    )

    env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
    eval_env = GymWrapper(eval_env, keys=['robot0_proprio-state', 'object-state'])

    if policy_train_selection == 1:
        train_reach_pick(env, eval_env)
    elif policy_train_selection == 2:
        train_pick_drop(env, eval_env)
    else:
        train_reach_drop(env, eval_env)
    
    print("Policy trained...")

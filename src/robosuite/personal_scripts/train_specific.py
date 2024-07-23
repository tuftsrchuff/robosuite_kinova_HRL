import numpy as np
import robosuite as suite
from robosuite.utils.mjmod import DynamicsModder
from robosuite.wrappers.behavior_cloning.hanoi_reach_pick import ReachPickWrapper
from robosuite.wrappers.behavior_cloning.hanoi_reach_drop import ReachDropWrapper
from robosuite.wrappers.behavior_cloning.hanoi_pick import PickWrapper
from robosuite.wrappers.behavior_cloning.hanoi_drop import DropWrapper
import os
from stable_baselines3.common.env_checker import check_env
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from robosuite import load_controller_config

controller_config = load_controller_config(default_controller='OSC_POSITION')

TRAINING_STEPS = 300000

def create_envs():
    # create environment instance
    env = suite.make(
        env_name="Hanoi",
        robots="Kinova3", 
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera="agentview",
        controller_configs=controller_config,
    )

    eval_env = suite.make(
        env_name="Hanoi",
        robots="Kinova3", 
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera="agentview",
        controller_configs=controller_config,
    )

    env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
    eval_env = GymWrapper(eval_env, keys=['robot0_proprio-state', 'object-state'])

    return env, eval_env

def train_reach_drop(env, eval_env):
    print("Training ReachDrop")
    #Wrap environment in the wrapper, try to train
    env = ReachDropWrapper(env)
    check_env(env)
    env = Monitor(env, filename=None, allow_early_resets=True)

    model = SAC.load("./models/ReachDrop/best_model.zip")

    eval_env = ReachDropWrapper(eval_env)
    eval_env = Monitor(eval_env, filename=None, allow_early_resets=True)

    # Define the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/ReachDrop',
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
        total_timesteps=TRAINING_STEPS,
        callback=eval_callback,
        progress_bar=True
    )

    # Save the model
    model.save(os.path.join('./models/reachdrop_sac'))

if __name__ == "__main__":
    env, eval_env = create_envs()
    train_reach_drop(env, eval_env)
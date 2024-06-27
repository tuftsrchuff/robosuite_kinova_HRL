import argparse
import numpy as np
import os
import robosuite as suite
import gymnasium
from stable_baselines3.common.env_checker import check_env
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite.wrappers.split_wrappers import PickWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robosuite import load_controller_config

controller_config = load_controller_config(default_controller='OSC_POSITION')


# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--timesteps', type=int, default=int(1e6), help='Number of timesteps to train for')
parser.add_argument('--task', type=str, default='pick', choices=['pick', 'reach', 'place'], help='Task to learn')
parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
args = parser.parse_args()

# Create the environment
env = suite.make(
    "PickPlace",
    robots="Kinova3",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    render_camera="agentview",
)
env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
env = PickWrapper(env, dense_reward=True)
check_env(env)
#env = DummyVecEnv([lambda: env])
# Normalize the environment
#env = VecNormalize(env, norm_obs=True, norm_reward=True)
env = Monitor(env, filename=None, allow_early_resets=True)

# Define the model
model = SAC(
    'MlpPolicy',
    env,
    learning_rate=args.lr,
    buffer_size=int(1e6),
    learning_starts=10000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    tensorboard_log='./logs/'
)

# Create the evaluation environment
eval_env = suite.make(
    "PickPlace",
    robots="Kinova3",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    render_camera="agentview",
)
eval_env = GymWrapper(eval_env, keys=['robot0_proprio-state', 'object-state'])
eval_env = PickWrapper(eval_env, dense_reward=True)
#eval_env = DummyVecEnv([lambda: eval_env])
#eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
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
    total_timesteps=args.timesteps,
    callback=eval_callback
)

# Save the model
model.save(os.path.join('./models/', args.task + '_sac'))
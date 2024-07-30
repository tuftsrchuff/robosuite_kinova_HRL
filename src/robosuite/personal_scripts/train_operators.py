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
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from robosuite import load_controller_config
import time

controller_config = load_controller_config(default_controller='OSC_POSITION')

TRAINING_STEPS = 1000000
ITERATION = 1

class BufferCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 0, path=None):
        super().__init__(verbose)
        # self.model = None 
        self.path = path

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        self.model.save_replay_buffer(self.path)
        return True


def train_reach_pick(env, eval_env):
    print("Training ReachPick")
    #Wrap environment in the wrapper, try to train
    env = ReachPickWrapper(env)

    #environment got invalid action dimension -- expected 8, got 4
    check_env(env)
    env = Monitor(env, filename=None, allow_early_resets=True, info_keywords=("is_success",))

    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=0.0003,
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
    eval_env = Monitor(eval_env, filename=None, allow_early_resets=True, info_keywords=("is_success"))

    buffer = BufferCallback(path=f'./models/ReachPick/{ITERATION}/buffer')

    # Define the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/ReachPick',
        log_path='./logs/',
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_on_new_best=buffer,
        verbose=1
    )

    # Train the model
    model.learn(
        total_timesteps=TRAINING_STEPS,
        callback=eval_callback,
        progress_bar=True
    )

    # Save the model
    model.save(os.path.join(f'./models/ReachPick/{ITERATION}/full/reachpick_sac'))
    model.save_replay_buffer(f"/models/ReachPick/{ITERATION}/full/reachpick_sac_replay_buffer")

def train_pick(env, eval_env):
    print("Training Pick")
    #Wrap environment in the wrapper, try to train
    env = PickWrapper(env)
    check_env(env)
    env = Monitor(env, filename=None, allow_early_resets=True, info_keywords=("is_success",))


    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=0.0003,
        buffer_size=int(1e6),
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log='./logs/Pick/'
    )

    eval_env = PickWrapper(eval_env)
    eval_env = Monitor(eval_env, filename=None, allow_early_resets=True, info_keywords=("is_success",))

    buffer = BufferCallback(path=f'./models/Pick/{ITERATION}/buffer')

    # Define the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./models/Pick/{ITERATION}/',
        log_path='./logs/Pick/',
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_on_new_best=buffer,
        verbose=1
    )

    # Train the model
    model.learn(
        total_timesteps=TRAINING_STEPS,
        callback=eval_callback,
        progress_bar=True
    )

    # Save the model
    model.save(os.path.join(f'./models/Pick/{ITERATION}/full/pick_sac'))
    model.save_replay_buffer(f"/models/Pick/{ITERATION}/full/pick_sac_replay_buffer")



def train_drop(env, eval_env):
    print("Training Drop")
    #Wrap environment in the wrapper, try to train
    env = DropWrapper(env)
    check_env(env)
    env = Monitor(env, filename=None, allow_early_resets=True, info_keywords=("is_success",))


    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=0.0003,
        buffer_size=int(1e6),
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log='./logs/Drop/'
    )

    eval_env = DropWrapper(eval_env)
    eval_env = Monitor(eval_env, filename=None, allow_early_resets=True, info_keywords=("is_success",))

    buffer = BufferCallback(path=f'./models/Drop/{ITERATION}/buffer')

    # Define the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./models/Drop/{ITERATION}/',
        log_path='./logs/Drop/',
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_on_new_best=buffer,
        verbose=1
    )


    # Train the model
    model.learn(
        total_timesteps=TRAINING_STEPS,
        callback=eval_callback,
        progress_bar=True
    )

    # Save the model
    model.save(os.path.join(f'./models/Drop/{ITERATION}/full/drop_sac'))
    model.save_replay_buffer(f"/models/Drop/{ITERATION}/full/drop_sac_replay_buffer")


def train_reach_drop(env, eval_env):
    print("Training ReachDrop")
    #Wrap environment in the wrapper, try to train
    env = ReachDropWrapper(env)
    check_env(env)
    env = Monitor(env, filename=None, allow_early_resets=True, info_keywords=("is_success",))

    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=0.0003,
        buffer_size=int(1e6),
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log='./logs/ReachDrop/'
    )

    eval_env = ReachDropWrapper(eval_env)
    eval_env = Monitor(eval_env, filename=None, allow_early_resets=True, info_keywords=("is_success",))

    buffer = BufferCallback(path=f'./models/ReachDrop/{ITERATION}/buffer')

    # Define the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./models/ReachDrop/{ITERATION}/',
        log_path='./logs/ReachDrop/',
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_on_new_best=buffer,
        verbose=1
    )

    # Train the model
    model.learn(
        total_timesteps=TRAINING_STEPS,
        callback=eval_callback,
        progress_bar=True
    )

    # Save the model
    model.save(os.path.join(f'./models/ReachDrop/{ITERATION}/full/reachdrop_sac'))
    model.save_replay_buffer(f"./models/ReachDrop/{ITERATION}/full/reachdrop_sac_replay_buffer")

def train_all():
    print("Training all")
    # env, eval_env = create_envs()
    # train_reach_pick(env, eval_env)
    
    # env, eval_env = create_envs()
    # train_pick(env, eval_env)

    env, eval_env = create_envs()
    train_drop(env, eval_env)

    env, eval_env = create_envs()
    train_reach_drop(env, eval_env)


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
        random_reset = True
    )

    eval_env = suite.make(
        env_name="Hanoi",
        robots="Kinova3", 
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera="agentview",
        controller_configs=controller_config,
        random_reset = True
    )

    env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
    eval_env = GymWrapper(eval_env, keys=['robot0_proprio-state', 'object-state'])

    return env, eval_env

if __name__ == "__main__":
    #Learn decomposed tasks for operators
    print("What policy would you like to train?")
    print("\t1. Reach-pick")
    print("\t2. Pick")
    print("\t3. Drop")
    print("\t4. Reach-drop")
    print("\t5. All")
    policy_train_selection = int(input())


    env, eval_env = create_envs()
    

    print(f"Training for {TRAINING_STEPS} steps")



    if policy_train_selection == 1:
        train_reach_pick(env, eval_env)
    elif policy_train_selection == 2:
        train_pick(env, eval_env)
    elif policy_train_selection == 3:
        train_drop(env, eval_env)
    elif policy_train_selection == 4:
        train_reach_drop(env, eval_env)
    else:
        train_all()
    
    print("Policy trained...")

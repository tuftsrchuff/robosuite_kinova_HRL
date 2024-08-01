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
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from robosuite import load_controller_config

controller_config = load_controller_config(default_controller='OSC_POSITION')

TRAINING_STEPS = 500000

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

def train_reach_drop(env, eval_env):
    print("Training ReachDrop")
    #Wrap environment in the wrapper, try to train
    env = ReachDropWrapper(env)
    check_env(env)
    env = Monitor(env, filename=None, allow_early_resets=True)

    model = SAC.load("./models/ReachDrop/1/full/reachdroptest_sac.zip", env=env)
    model.load_replay_buffer("./models/ReachDrop/1/full/reachdrop_sac_replay_buffer.pkl")
    #Load replay buffer
    print("Loaded model and replay buffer")
    print(f"Training for {TRAINING_STEPS}")

    eval_env = ReachDropWrapper(eval_env)
    eval_env = Monitor(eval_env, filename=None, allow_early_resets=True)


    buffer = BufferCallback(path=f'./models/ReachDrop/1/buffer')

    # Define the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/ReachDrop/1/',
        log_path='./logs/ReachDrop',
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
    model.save(os.path.join('./models/ReachDrop/1/reachdrop_sac_updated'))
    model.save_replay_buffer('./models/ReachDrop/1/reach_drop_buffer_updated')

def train_reach_drop(env, eval_env):
    print("Training Drop")
    #Wrap environment in the wrapper, try to train
    env = ReachDropWrapper(env)
    check_env(env)
    env = Monitor(env, filename=None, allow_early_resets=True)

    model = SAC.load("./models/Drop/2/full/drop_sac.zip", env=env)
    model.load_replay_buffer("./models/Drop/2/full/drop_sac_replay_buffer.pkl")
    #Load replay buffer
    print("Loaded model and replay buffer")
    print(f"Training for {TRAINING_STEPS}")

    eval_env = DropWrapper(eval_env)
    eval_env = Monitor(eval_env, filename=None, allow_early_resets=True)


    buffer = BufferCallback(path=f'./models/Drop/2/buffer')

    # Define the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/Drop/2/',
        log_path='./logs/Drop',
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
    model.save(os.path.join('./models/Drop/2/full/drop_sac_updated'))
    model.save_replay_buffer('./models/Drop/2/full/drop_buffer_updated')


if __name__ == "__main__":
    env, eval_env = create_envs()
    train_reach_drop(env, eval_env)
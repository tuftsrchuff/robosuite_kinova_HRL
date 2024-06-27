import numpy as np
import robosuite as suite
from robosuite.wrappers.split_wrappers import PickWrapper
import time

"""# create environment instance
env = suite.make(
    env_name="PickPlace", # try with other tasks like "Stack" and "Door"
    robots="Kinova3",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    render_camera="frontview",
)
#'frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand'

# wrap the environment with the PickAndPlaceWrapper
env = PickWrapper(env, dense_reward=True)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display"""

pick_wrapper = PickWrapper(dense_reward=True)

obs = pick_wrapper.reset()

for i in range(1000):
    action = np.random.uniform(size=4) # sample random action from a uniform distribution
    obs, reward, done, info = pick_wrapper.step(action)
    pick_wrapper.render()  # render on display
    time.sleep(1)
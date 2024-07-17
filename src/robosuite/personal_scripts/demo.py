import numpy as np
import robosuite as suite
from robosuite.utils.mjmod import DynamicsModder

# create environment instance
env = suite.make(
    env_name="Hanoi", # try with other tasks like "Stack" and "Door"
    robots="Kinova3",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

#env.robots stores array of robot objects

action = np.random.randn(env.robots[0].dof) # sample random action
data = env.step(action)  # take action in the environment
print(data)

for i in range(100):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display

# env.robots[0].set_joint_attribute("friction loss", np.ones(len(env.robots[0].joint_indexes)))
# print("Manually setting friction")

# modder = DynamicsModder(sim=env.sim, random_state=np.random.RandomState(5))
# modder.mod(geom_name, "friction", [2.0, 0.2, 0.04])  

# modder.update()                                                   # make sure the changes propagate in sim

# for i in range(100):
#     action = np.random.randn(env.robots[0].dof) # sample random action
#     obs, reward, done, info = env.step(action)  # take action in the environment
#     env.render()  # render on display
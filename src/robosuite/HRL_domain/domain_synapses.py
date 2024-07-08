import warnings
import robosuite as suite

def create_env(env_id, render=False, seed=None):
    '''
    This method is responsible for generating a new environment for the experiment. It takes an environment ID, a list of novelties, 
    a reset location and direction, a render flag, and a training flag as parameters. 

    If the environment does not exist or the novelty list has changed, it closes the existing environment (if any) and creates a new one. 
    It sets the render, reset location and direction, and environment ID attributes of the instance. 
    It determines the locality of the environment based on the novelties. If there are multiple novelties or a global novelty, the locality 
    is set to "global". If there is a single local novelty, the locality is set to "local".
    It then injects the novelties into the environment by creating a new instance of the novelty wrapper for each novelty and adding it to 
    the environment. 
    Finally, it seeds the environment and sets the environment and novelty list attributes of the instance.
    '''
    print(f"Creating env ID: {env_id}, render {render}, seed {seed}")


    # create environment instance
    env = suite.make(
        env_name="Hanoi", # try with other tasks like "Stack" and "Door"
        robots="Kinova3",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    return env

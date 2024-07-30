from planner import *
from executor_noHRL import Executor
from robosuite.HRL_domain.domain_synapses import *
from robosuite.wrappers.gym_wrapper import GymWrapper
import time
from learner import Learner

def decomposeAction(action):
    print("Decomposing")
    action = action.lower()
    components = action.split(' ')
    base_action = components[0]
    toMove = components[1]
    destination = components[3]
    return base_action, toMove, destination
    

def executeAction(base_action, toMove, destination, env):
    # obs = np.concatenate((obs, self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]][:3]))
    #ignore_done=False - used in the environment which gym wrapper calls, maybe pass into 
    exec_env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])


    #Try to perform operator then pass back success or not on operator
    print(f"Taking action {base_action}")
    print(f"Reach-pick {toMove} to {destination}")
    executor = Executor(exec_env, 'reach_pick')
    success = executor.execute_policy(symgoal=toMove)
    print(f"Success: {success}")
    if not success:
        return success, 'reach_pick'

    #Terminated environment, use base env and rewrap?
    print(f"Pick {toMove}")
    # exec_env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
    executor = Executor(exec_env, 'pick')
    success = executor.execute_policy(symgoal=toMove)
    print(f"Success: {success}")
    if not success:
        return success, 'pick'

    print(f"Reach-drop {destination}")
    # exec_env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
    executor = Executor(exec_env, 'reach_drop')
    success = executor.execute_policy(symgoal=[toMove,destination])
    print(f"Success: {success}")
    if not success:
        return success, 'reach_drop'

    print(f"Dropping {destination}")
    # exec_env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
    executor = Executor(exec_env, 'drop')
    success = executor.execute_policy(symgoal=[toMove,destination])
    print(f"Success: {success}")
    if not success:
        return success, 'drop'
    else:
        return success, None


def call_learner(operator, env):
    print(f"Learning new operator {operator}")
    learner = Learner(env, operator)
    learner.learn()
    print("New operator learned")
    time.sleep(5)


if __name__ == "__main__":
    pddl_dir = "../PDDL"
    domain_dir = "Domains"
    problem_dir = "Problems"
    domain = "domain"
    problem = "problem"

    domain_path = pddl_dir + os.sep + domain + ".pddl"
    problem_path = pddl_dir + os.sep + problem + ".pddl"
    print("Solving tower of Hanoi task")

    env = suite.make(
        env_name="Hanoi",
        robots="Kinova3", 
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        # render_camera="agentview",
        controller_configs=controller_config,
        ignore_done=True
    )

    #Full plan execution with normal environment

    plan, game_action_set = call_planner(domain_path, problem_path)
    for action in plan:
        base_action, toMove, destination = decomposeAction(action)
        success, operator = executeAction(base_action, toMove, destination, env)
        if not success:
            call_learner(operator, env)
            success, operator = executeAction(base_action, toMove, destination, env)


    #Novelty injection
    env = suite.make(
            env_name="DoorNovelty",
            robots="Kinova3", 
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            render_camera="agentview",
            controller_configs=controller_config
        )
    plan, game_action_set = call_planner(domain_path, problem_path)
    for action in plan:
        base_action, toMove, destination = decomposeAction(action)
        executeAction(base_action, toMove, destination, env)

from planner import *
from executor_noHRL import Executor
from robosuite.HRL_domain.domain_synapses import *
from robosuite.wrappers.gym_wrapper import GymWrapper
import time

def decomposeAction(action):
    print("Decomposing")
    action = action.lower()
    components = action.split(' ')
    base_action = components[0]
    toMove = components[1]
    destination = components[3]
    return base_action, toMove, destination
    
    

def executeAction(base_action, toMove, destination, env):

    print(f"Taking action {base_action}")
    #Post condition can be looked at here - toMove should be on destination
    print(f"Reach-pick {toMove} to {destination}")
    # obs = np.concatenate((obs, self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]][:3]))
    exec_env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
    executor = Executor(exec_env, 'reach_pick')
    success = executor.execute_policy(symgoal=toMove)
    if not success:
        print("Reach pick failed")
        return False
    # time.sleep(5)

    #Terminated environment, use base env and rewrap?
    print(f"Pick {toMove}")
    # exec_env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
    executor = Executor(exec_env, 'pick')
    success = executor.execute_policy(symgoal=toMove)
    if not success:
        print("Pick failed")
        return False
    # time.sleep(5)

    print(f"Reach-drop {destination}")
    # exec_env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
    executor = Executor(exec_env, 'reach_drop')
    success = executor.execute_policy(symgoal=[toMove,destination])
    if not success:
        print("Reach drop failed")
        return False
    # time.sleep(5)

    print(f"Dropping {destination}")
    # exec_env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
    executor = Executor(exec_env, 'drop')
    success = executor.execute_policy(symgoal=[toMove,destination])
    if not success:
        print("Drop failed")
        return False
    else:
        print(f"Action {base_action} {toMove} onto {destination} finished successfully")
        return True





if __name__ == "__main__":
    pddl_dir = "../PDDL"
    domain_dir = "Domains"
    problem_dir = "Problems"
    domain = "domain_v2"
    problem = "problem"

    domain_path = pddl_dir + os.sep + domain + ".pddl"
    problem_path = pddl_dir + os.sep + problem + ".pddl"
    print("Solving tower of Hanoi task")
    env = create_env("ReachPick")

    
    plan, game_action_set = call_planner(domain_path, problem_path)
    print(plan)
    # plan = ['MOVE CUBE1 CUBE2 PEG2', 'MOVE CUBE2 CUBE3 PEG3', 'MOVE CUBE1 PEG2 CUBE2', 'MOVE CUBE3 PEG1 PEG2', 'MOVE CUBE1 CUBE2 PEG1', 'MOVE CUBE2 PEG3 CUBE3', 'MOVE CUBE1 PEG1 CUBE2']

    # for action in plan:
    #     base_action, toMove, destination = decomposeAction(action)
    #     success = executeAction(base_action, toMove, destination, env)
    #     if not success:
    #         print("Plan failed")
    #         break
    # print(plan)
    #First call planner and return the plan
    #Plan - ['MOVE D1 D2 PEG2', 'MOVE D2 D3 PEG3', 'MOVE D1 PEG2 D2', 'MOVE D3 D4 PEG2', 'MOVE D1 D2 D4', 'MOVE D2 PEG3 D3', 'MOVE D1 D4 D2', 'MOVE D4 PEG1 PEG3', 'MOVE D1 D2 D4', 'MOVE D2 D3 PEG1', 'MOVE D1 D4 D2', 'MOVE D3 PEG2 D4', 'MOVE D1 D2 PEG2', 'MOVE D2 PEG1 D3', 'MOVE D1 PEG2 D2']
    
    #Load in the domain synapse information here - will have obj body mapping
    
    #Run executor until it's done, done state can be checked in detector? If block 2 on 3, etc
        # MOVE D1 D2 PEG2
        # Reach-pick d1
        # Pick d1
        # reach drop peg2
        # drop peg2
    
    #Decompose task, reach-pick d1, pick d1, reach-drop peg2, drop peg2
        

    #Then create simple executor that just executes full policy
        #What would the pre and post conditions be?
            #Specified in the plan itself
        #Check pre and post conditions
    

    #Append the observation with the goal

    #Execute full policy for plan step

    #Optimal if you could watch this all happen
        #Success/failure message
        #Is the disk on the desired block? Should be able to see it happen
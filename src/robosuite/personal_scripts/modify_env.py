from planner import *
from executor_noHRL import Executor
from robosuite.HRL_domain.domain_synapses import *
from robosuite.wrappers.gym_wrapper import GymWrapper
import time
from robosuite.utils.mjmod import DynamicsModder


def executeAction(toMove, env):
    exec_env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
    executor = Executor(exec_env, 'reach_pick')
    done = executor.execute_policy(symgoal=toMove)


    

if __name__ == "__main__":
    print("Executing standard reach_pick operator...")
    env = create_env("ReachPick")
    # time.sleep(5)
    # print("\n\n\n")
    # print(vars(env.robots[0].robot_model.joints[0]))
    # time.sleep(15)

    def print_params():
        print(f"cube mass: {env.sim.model.body_mass[cube_body_id]}")
        print(f"cube frictions: {env.sim.model.geom_friction[cube_geom_ids]}")
        print()

    joints = env.robots[0].robot_model.joints
    print(joints)

    # modder = DynamicsModder(sim=env.sim, random_state=np.random.RandomState(5))
    # # modder.mod(env.robot.joint, "damping", 0.8)                                # make the joint stiff
    # for joint in env.robots[0].robot_model.joints:
    #     modder.mod(joint, "stiffness", 5.0)          # greatly increase the friction

    # modder.update()
    # print("Updated damping in joint")
    # time.sleep(5)


    executeAction("cube1", env)
    time.sleep(10)
    env.reset()



    print("Executing pick with novelty injection")
    print(env.sim)
    joints = env.robots[0].robot_model.joints
    print(joints)
    modder = DynamicsModder(sim=env.sim, random_state=np.random.RandomState(5))
    modder.mod(env.robot.joint, "stiffness", 5.0)                                # make the joint stiff
    for joint in env.robot.joint:
        modder.mod(joint, "stiffness", [2.0, 0.2, 0.04])           # greatly increase the friction
    modder.update()                                                   # make sure the changes propagate in sim

    # Print out modified parameter values
    print("MODIFIED VALUES")
    print_params()

    # We can also restore defaults (original values) at any time
    modder.restore_defaults()

    # Print out restored initial parameter values
    print("RESTORED VALUES")
    print_params()



    #TODO: Update this modder to properly update friction on all joints
    self.modder.mod(geom_name, "friction", [2.0, 0.2, 0.04])  
    self.modder.update()  
    
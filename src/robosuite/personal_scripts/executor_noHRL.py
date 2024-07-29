
'''
# This files implements the structure of the executor object used to execute the hierarchical policies
'''
from stable_baselines3 import SAC, PPO
from robosuite.HRL_domain.detector import Detector
from robosuite.HRL_domain.domain_synapses import *
import numpy as np
import time

class Executor():
    def __init__(self, env, operator):
        self.env = env
        self.detector = Detector(env)
        self.operator = operator

        populateExecutorInfo(env)

    
    def execute_policy(self,
                       symgoal = None):


        done = False
        rew_eps = 0
        step_executor = 0
        model = SAC.load(executors[self.operator])

        #Base action
        base_action = np.zeros(len(self.env.action_space.sample()))
        obs, _, _, _, _ = self.env.step(base_action)

        #addGoal - generic

        obs = addGoal(obs, symgoal, self.env, self.operator)

        

        # #Adds 3D goal from symgoal
        # if self.operator in ["reach_pick", "pick", "reach_drop"]:
        #     obs = np.concatenate((obs, self.env.sim.data.body_xpos[self.obj_mapping[symgoal]][:3]))
        # else:
        #     obs = np.concatenate((obs, self.env.sim.data.body_xpos[self.obj_mapping[symgoal[1]]][:3]))

        Beta = termination_indicator(self.operator)
        terminated = False
        print(symgoal)

        #Need to pass in initial observation somehow        
        while not done and not terminated:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)

            #addGoal
            obs = addGoal(obs, symgoal, self.env, self.operator)
            step_executor += 1
            rew_eps += reward
            done = Beta(self.env, symgoal)
            self.env.render()

            if step_executor > 1000:
                done = True

        print(f"Terminated: {terminated}")
        print(f"Done: {done}")
        # comparing execution effects to expected effects
        new_state = self.detector.get_groundings(self.env)


        expected_effects_keys = effects(self.operator, symgoal)
        print(expected_effects_keys)

        #Compare looks at all predicates in new state and checks if it exists in grounded
        #predicates in old state, if not adds it to the execution_effects
        execution_effects = []
        for effect in expected_effects_keys:
            execution_effects.append(new_state[effect])

        expected_effects = effect_mapping[self.operator]
        print(f"Expected effects {expected_effects}")
        print(f"Execution efects {execution_effects}")


        success = (all(x in execution_effects for x in expected_effects))
        if success:
            return True
        else:
            print(f"{self.operator} failed...")
            return False

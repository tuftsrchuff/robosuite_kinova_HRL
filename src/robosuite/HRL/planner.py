'''
# This file is the connection to the planner and PDDL/HDDL knowledge of RAPidL.
# It implements the hddl problem generator and planner function.

Important References

'''
import os
import copy
import subprocess
from HDDL.generate_hddl import *
import domain_synapses

hddl_dir = "HDDL"

def call_planner(domain, problem, structure="hddl"):
    '''
        Given a domain and a problem file
        This function return the ffmetric Planner output.
        In the action format
    '''
    global applicator
    applicator = domain_synapses.applicator
    #hddl_dir = "HDDL"
    if structure == "hddl":
        # print("./lilotane/build/lilotane "+hddl_dir+os.sep+domain+".hddl "+ hddl_dir+os.sep+problem+".hddl -v=0 -cs")
        run_script = "./lilotane/build/lilotane "+hddl_dir+os.sep+domain+".hddl "+ hddl_dir+os.sep+problem+".hddl -v=0 -cs"# | cut -d' ' -f2- | sed 1,2d | head -n -2" # > + sub_plan_name
        output = subprocess.getoutput(run_script)
        #print(output)
        if "Unsolvable" in output:
            #print("Plan not found with Lilotane! Error: {}".format(
            #    output))
            return False, False
        elif "[glucose4]" in output:
            #print("Plan not found with Lilotane! Error: {}".format(
            #    output))
            return False, False
        try:
            output = output.rsplit('==>', 1)[1]
            output = output.rsplit('root', 1)[0]
        except Exception as e:
            print("The planner failed because of: {}.\nThe output of the planner was:\n{}".format(e, output))

        plan, game_action_set = _output_to_plan(output, structure=structure)
        return plan, game_action_set

def _output_to_plan(output, structure):
    '''
    Helper function to perform regex on the output from the planner.
    ### I/P: Takes in the ffmetric output and
    ### O/P: converts it to a action sequence list.
    '''
    if structure == "hddl":
        action_set = []
        for action in output.split("\n")[1:-1]:
            #print(' '.join(action.split(" ")[1:]))
            action_set.append(' '.join(action.split(" ")[1:]))
        # print("n------------------------------The plan is: ----------------------------\n")
        # print(action_set)
        
        # convert the action set to the actions permissable in the domain
        game_action_set = copy.deepcopy(action_set)

        for i in range(len(game_action_set)):
            game_action_set[i] = applicator[game_action_set[i].split(" ")[0]]
        for i in range(len(game_action_set)):
            for j in range(len(game_action_set[i])):
                if game_action_set[i][j] in applicator.keys():
                    game_action_set[i][j] = applicator[game_action_set[i]]
        return action_set, game_action_set

def generate_hddls(state, task, new_item=None, filename: str = "problem"):
    hddl_dir = "HDDL"
    os.makedirs(hddl_dir, exist_ok = True)
    generate_prob_hddl(hddl_dir, state, task, filename=filename)
    if new_item is not None:
        print("new item adding to the domain file = ", new_item)
        generate_domain_hddl(hddl_dir, state, new_item)
'''
This function generates the HDDLs from the current environment instance
### I/P environment object
### O/P returns hddl names if DDL generated successfully, else returns false.
'''
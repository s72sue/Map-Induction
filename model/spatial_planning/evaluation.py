import argparse

import os
import sys

if("PYTHONPATH" not in os.environ.keys()):
    sys.path.insert(0,'/om2/user/curtisa/spatial_planning')

from src.agents import POMCP, \
                       RandomPolicy, \
                       LandmarkPolicy, \
                       LandmarkRewardPolicy

import copy
import pickle
from src.utils import readMap, write_to_minio
import time
import os
import numpy as np

AGENTS = {"pomcp_simple": POMCP,
          "pomcp_ssp": POMCP,
          "pomcp_mle": POMCP,
          "random_policy": RandomPolicy,
          "landmark_policy": LandmarkPolicy,
          "landmark_reward_policy": LandmarkRewardPolicy
          }

TASKS = {
    "small_simple_room": ["sim/small_simple_room"],
    "small_stim_walled": ["sim/small_stim_walled"],
    "stim1": ["sim/stim1"],
    "stim_walled": ["sim/stim_walled"],
    "four_room": ["sim/four_room"],
    "one_side": ["sim/one_side"],
    "chain": ["sim/chain"],
    "lattice": ["sim/lattice"],
    "two_room_choice_right": ["sim/two_room_choice_right"],

    "test1": ["sim_exp1/test1"],
    "test1_Reflect": ["sim_exp1/test1_Reflect"],
    "test2": ["sim_exp1/test2"],
    "test2_Reflect": ["sim_exp1/test2_Reflect"],
    "test3": ["sim_exp1/test3"],
    "test3_Reflect": ["sim_exp1/test3_Reflect"],
    "test4": ["sim_exp1/test4"],
    "test4_Reflect": ["sim_exp1/test4_Reflect"],
    "test5": ["sim_exp1/test5"],
    "test5_Reflect": ["sim_exp1/test5_Reflect"],
    "test5_Improved": ["sim_exp1/test5_Improved"],
    "test5_Reflect_Improved": ["sim_exp1/test5_Reflect_Improved"],
    "test6": ["sim_exp1/test6"],
    "test6_Reflect": ["sim_exp1/test6_Reflect"],

    "test7": ["stims_exp3/test7"],
    "test8": ["stims_exp3/test8"],
    "test9": ["stims_exp3/test9"],
    "test10": ["stims_exp3/test10"],
    "test11": ["stims_exp3/test11"],
    "test12": ["stims_exp3/test12"],

    "discovery": ["sim/two_room_choice_left",
                  "sim/two_room_choice_right",
                  "sim/two_room_choice_middle"]
}

REPLAN_STRATS = ["every_step", "mpc"]
OBSERVATION_MODES = ["fixed_radius", "fixed_radius_med", "line_of_sight", "directional_line_of_sight"]


def get_single_result(agent_name, task_name, exp_dict):
    stim_filenames = ["./stim/"+env_name+".txt" for env_name in TASKS[task_name]]
    stims = [readMap(stim_fn) for stim_fn in stim_filenames]
    Agent = AGENTS[agent_name]
    return Agent([np.array(stim) for stim in stims], **exp_dict).run()


def save_result(save_dict, spec_dict, save_name):
    if(save_dict['write_external']):
        write_to_minio(save_dict, save_name)
        write_to_minio(spec_dict, "spec_"+str(save_name))
    else:
        if not os.path.exists(save_dict['save_dir']):
            os.makedirs(save_dict['save_dir'])
        filename = save_dict['save_dir'] + save_name + ".pkl"
        pickle.dump(save_dict, open(filename, 'wb'))


def update_env_dict(exp_dict, states, observed_states, actions, rewards, info):
    exp_dict.update({
        "states": states,
        "observed_states": observed_states,
        "actions": actions,
        "rewards": rewards,
        "info": info
        })
    return exp_dict


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--task-name',  choices=TASKS.keys(), default='test7', help='')
parser.add_argument('-exp', '--exp-name', default='debug_multi_2', help='')
parser.add_argument('-we', '--write-external', action='store_true', help='') # For cloud experiments
parser.add_argument('-a', '--agent-name',  choices=list(AGENTS.keys()), default='pomcp_ssp', help='')
parser.add_argument('-sd', '--search-depth', type=int, default=20, help='')
parser.add_argument('-tq', '--tree-queries', type=int, default=500, help='')
parser.add_argument('-df', '--discount-factor', type=float, default=0.90, help='')
parser.add_argument('-op', '--optimism', type=float, default=0.000001, help='')
parser.add_argument('-o', '--observation_mode', choices=OBSERVATION_MODES, default="directional_line_of_sight", help='')
parser.add_argument('-it', '--num_iterations', type=int, default=300, help='')
parser.add_argument('-s', '--save-dir',  default='./results/', help='')
parser.add_argument('-rp', '--replan-strat',
                    default='every_step',
                    choices=REPLAN_STRATS,
                    help='select model for segmentation. {}'.format(' | '.join(REPLAN_STRATS)))

if __name__ == "__main__":

    args = parser.parse_args()
    exp_dict = dict(vars(args))
    result_dict = update_env_dict(copy.copy(exp_dict), *get_single_result(args.agent_name, args.task_name, exp_dict))
    save_result(result_dict, exp_dict, str(time.time()))


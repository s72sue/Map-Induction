import argparse

import os
import sys

if("PYTHONPATH" not in os.environ.keys()):
    sys.path.insert(0,'/om2/user/curtisa/spatial_planning')


import copy
import pickle
from src.utils import readMap, write_to_minio, isolate_rooms
from src.params import *
import time
import os
import numpy as np
          
def get_single_result(agent_name, task_name, exp_dict):
    stim_filenames = ["./stim/{}.txt".format(env_name) for env_name in TASKS[task_name]]
    room_filenames = ["./stim/{}.txt".format(env_name) for env_name in TASKS[task_name+"_room"]]
    stims = [readMap(stim_fn) for stim_fn in stim_filenames]
    room_keys = [isolate_rooms(readMap(stim_fn)) for stim_fn in room_filenames]
    Agent = AGENTS[agent_name]
    exp_dict["optimism"] = OPT[task_name]
    return Agent([np.array(stim) for stim in stims], 
                 room_keys=[np.array(room) for room in room_keys], 
                 **exp_dict).run()


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

env = "test10"
mode = "pomcp_ssp"

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--task-name',  choices=TASKS.keys(), default=env, help='')
parser.add_argument('-exp', '--exp-name', default='{}_los_real_2_{}'.format(env, mode), help='')
parser.add_argument('-we', '--write-external', action='store_true', help='') # For cloud experiments
parser.add_argument('-a', '--agent-name',  choices=list(AGENTS.keys()), default=mode, help='')
parser.add_argument('-sd', '--search-depth', type=int, default=20, help='')
parser.add_argument('-tq', '--tree-queries', type=int, default=1000, help='')
parser.add_argument('-df', '--discount-factor', type=float, default=0.90, help='')
parser.add_argument('-op', '--optimism', type=float, default=0.00002, help='')
parser.add_argument('-o', '--observation_mode', choices=OBSERVATION_MODES, default="room", help='')
parser.add_argument('-it', '--num_iterations', type=int, default=5000, help='')
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


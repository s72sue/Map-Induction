import os
import sys

if("PYTHONPATH" not in os.environ.keys()):
    sys.path.insert(0,'/om2/user/curtisa/spatial_planning')

from src.human_exp_database import HumanExperimentResults
from src.consts import *
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
import pickle5
from matplotlib import cm
from matplotlib.dates import date2num
import datetime
import math
from src.utils import *
from src.agents import Agent, POMCP
from collections import defaultdict
import evaluation
from src.utils import readMap, write_to_minio
from src.params import *

evaluation.parser.add_argument('-p', '--participant', help='')
args = evaluation.parser.parse_args()
exp_dict = dict(vars(args))

p1 = args.participant[1:]

with open("./notebooks/exp_results/human_experiment_results_{}.pkl".format(p1), 'rb') as handle:
    human_exps = pickle5.load(handle)


def bound(y, x, grid):
    if(y>=np.array(grid).shape[0]):
        y = np.array(grid).shape[0]-1
    elif(y<0):
        y=0

    if(x >= np.array(grid).shape[1]):
        x = np.array(grid).shape[1]-1
    elif(x<0):
        x=0
    return (y, x)

class HumanAgent(POMCP):
    def __init__(self, stims, human_actions, room_keys, *args, **kwargs):
        super(HumanAgent, self).__init__(stims, room_keys=room_keys, **kwargs)
        self.human_actions = human_actions
        self.i=0
        self.alternate_policy=True

    def get_alternate_actions(self, current_observed_state, current_observation, previous_reward, run_infos):
        print("Human Traj Action {}/{}".format(self.i, len(self.human_actions)))
        action = self.human_actions[self.i]
        self.i+=1
        if(self.i == len(self.human_actions)):
            self.term=True

        return [action]


cmap =  cm.get_cmap("plasma").colors
def get_agent(s):
    return (list(np.argwhere(s==AGENT_UP)) + list(np.argwhere(s==AGENT_DOWN)) + list(np.argwhere(s==AGENT_LEFT)) + list(np.argwhere(s==AGENT_RIGHT)))

# Heatmaps
# for tmap, exps in human_exps.datas.items():
#     for p1 in human_exps.datas[tmap].keys():
#         for mode in ["pomcp_simple", "pomcp_mle", "pomcp_ssp"]:

tmap = args.task_name
observed_state, real_states,  actions, _, _ = human_exps.datas[tmap][p1]

# Run the simulation
exp_dict['num_iterations'] = len(real_states)
# exp_dict['num_iterations'] = 
mode = exp_dict['agent_name']
print(exp_dict)

offsets = {
    "test1":[0, 0],
    "test2":[0, 0],
    "test3":[0, 0],
    "test4":[0, 0],
    "test5":[0, 0],
    "test6":[0, 0],
    "test7":[1, 1],
    "test8":[-4, 1],
    "test9":[1, 1],
    "test10":[3, 1],
    "test11":[16, 1],
    "test12":[2, 0]
}

# Wait until person pases threshold
start_agent = get_agent(real_states[0])[0]


while(True):
    print(get_agent(real_states[0])[0][0] - start_agent[0])
    if(get_agent(real_states[0])[0][0] - start_agent[0] <= offsets[exp_dict["task_name"]][0]):
        offsets[exp_dict["task_name"]][0] =  - ((get_agent(real_states[0])[0][0] - start_agent[0]) - offsets[exp_dict["task_name"]][0])
        offsets[exp_dict["task_name"]][1] =  (get_agent(real_states[0])[0][1] - start_agent[1]) + offsets[exp_dict["task_name"]][1]
        break
    else:
        real_states = real_states[1:]
        actions = actions[1:]
        observed_state = observed_state[1:]
 

# # Save the policy
agent_name, task_name = exp_dict["agent_name"], exp_dict["task_name"]
stim_filenames = ["./stim/{}.txt".format(env_name) for env_name in TASKS[task_name]]
room_filenames = ["./stim/{}.txt".format(env_name) for env_name in TASKS[task_name+"_room"]]
stims = [readMap(stim_fn) for stim_fn in stim_filenames]
room_keys = [isolate_rooms(readMap(stim_fn)) for stim_fn in room_filenames]
write_mapcode_int("iso.txt", np.array(room_keys[0]))

Agent = AGENTS[agent_name]
exp_dict["optimism"] = OPT[task_name]    

stim_agent = get_agent(stims[0])
stims[0][stim_agent[0][0], stim_agent[0][1]] = EMPTY
stims[0][stim_agent[0][0]-offsets[exp_dict["task_name"]][0], stim_agent[0][1]+offsets[exp_dict["task_name"]][1]] = AGENT


human_agent = HumanAgent([stims[0]], actions, room_keys=[np.array(room) for room in room_keys], **exp_dict)
states, observed_states, actions, rewards, run_infos = human_agent.run()

# Read the human experiment
traj_rooms = []
traj_els = []
for real_state in states:
    traj_el = get_agent(real_state)[0]
    traj_els.append(traj_el)
    traj_el_room = np.array(room_keys[0])[traj_el[0], traj_el[1]]
    traj_rooms.append(traj_el_room)


single_probs = []

for ti, traj_el in enumerate(traj_rooms[:-1]):
    if(ti+1 >= len(run_infos)):
        break
    run_info = run_infos[ti+1]
    policy = run_info['policy']

    tii = ti
    while(traj_rooms[tii] == traj_el and tii < (len(traj_rooms)-1)):
        tii += 1
        next_room = traj_rooms[tii]
        

    tree_paths = policy
    new_room_indices = []

    for tree_path in tree_paths:
        for si, state in enumerate(tree_path):
            if state[2] is None:
                obs = real_states[ti]
                apos = get_agent(obs)[0]
            else:
                obs = state[2]
                apos = (obs[0]-1, obs[1]-1)

            room_index = room_keys[0][apos[0]][apos[1]]

            if(room_index != traj_el and si==len(tree_path)-1):
                new_room_indices.append((room_index, state[1]))

    if(len(new_room_indices)>0):
        total_probs = defaultdict(lambda:0)
        for new_room_index in new_room_indices:
            total_probs[new_room_index[0]]+=(new_room_index[1]+1)

        print(total_probs)
        print("next room: {}->{}".format(str(traj_el), str(next_room)))
        single_probs.append(total_probs[next_room]/float(sum(total_probs.values())))
    else:
        single_probs.append(0)
# human_cross_env_probs[tmap][p1][mode] = single_probs
save_name = '{}_results_{}_{}'.format(mode, tmap, p1)
with open(save_name+".pkl", 'wb') as handle:
    pickle5.dump(single_probs, handle, protocol=pickle5.HIGHEST_PROTOCOL)
write_to_minio(single_probs, save_name)
print(single_probs)    



    
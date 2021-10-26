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


NEW_WALL = 47
evaluation.parser.add_argument('-p', '--participant', help='')
args = evaluation.parser.parse_args()
exp_dict = dict(vars(args))

p1 = args.participant[1:]

with open("./notebooks/human_{}_results.pkl".format(p1), 'rb') as handle:
    human_exps = pickle5.load(handle)

class HumanAgent(POMCP):
    def __init__(self, stims, human_actions, *args, **kwargs):
        super(HumanAgent, self).__init__(stims, **kwargs)
        self.human_actions = human_actions
        self.i=0
        self.alternate_policy=True

    def get_alternate_actions(self, current_observed_state, current_observation, previous_reward, run_infos):
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
print("p1, tmap")
print(tmap)
print(p1)
observed_state, real_states,  actions, _, _ = human_exps.datas[tmap][p1]

# Read room naming map from different folder
room_map_fn = "stim/sim_exp1_rooms/{}.txt".format(tmap)
room_map = readMap(room_map_fn)

# Read the human experiment
traj_rooms = []
traj_els = []
for real_state in real_states:
    traj_el = get_agent(real_state)[0]
    traj_els.append(traj_el)
    traj_el_room = room_map[traj_el[0]][traj_el[1]]
    traj_rooms.append(traj_el_room)
    
print(traj_rooms)

# Run the simulation
exp_dict['num_iterations'] = len(real_states)
# exp_dict['num_iterations'] = 
mode = exp_dict['agent_name']

print(exp_dict)

# # Save the policy
human_agent = HumanAgent([real_states[0]], actions,  **exp_dict)
_, _, _, _, run_infos = human_agent.run()

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
        initial_room_index = None
        tree_path_pos = []
        for si, state in enumerate(tree_path):
            if state[2] is None:
                obs = real_states[ti]
                apos = get_agent(obs)[0]
                initial_room_index =  room_map[apos[0]][apos[1]]
            else:
                obs = state[2]
                apos = (obs[0]-1, obs[1]-1)

            room_index = room_map[apos[0]][apos[1]]

            if(room_index != NEW_WALL and room_index != initial_room_index and si==len(tree_path)-1):
                new_room_indices.append((room_index, state[1]))
            tree_path_pos.append(room_index)

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



    
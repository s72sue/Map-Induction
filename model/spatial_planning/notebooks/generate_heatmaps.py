
import pickle
from os import listdir
from os.path import isfile, join
import imageio
import time
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from PIL import Image
import copy
from matplotlib import cm
from src.utils import *
from src.map_grid import *
from src.programs import *
from src.matcher import *
from src.proposal import *
from src.inference import *
from src.params import *
import minio
import matplotlib.pyplot as plt
from src.agents import Agent, POMCP
import evaluation


class HeatmapAgent(POMCP):
    def __init__(self, stims, human_actions, room_keys, *args, **kwargs):
        super(HeatmapAgent, self).__init__(stims, room_keys=room_keys, **kwargs)
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


exp_version ="exp2"
remote=False
if(exp_version == "exp1"):
    exp=["test1", 
#           "test1_Reflect", 
          "test2", 
#           "test2_Reflect", 
          "test3", 
#           "test3_Reflect", 
          "test4", 
#           "test4_Reflect", 
          "test5",
#           "test5_Reflect",
          "test6", 
#           "test6_Reflect"
        ]
elif(exp_version == "exp2"):
    exp = [
        # "test7", 
        # "test8", 
        # "test9", 
        # "test10", 
        "test11", 
        # "test12"
    ]

    
modes = ["pomcp_mle"]

args = evaluation.parser.parse_args()
exp_dict = dict(vars(args))
exp_dict['observation_mode'] = "line_of_sight"

all_experiment_specs = load_all_experiment_specs_minio()
for mode in modes:
    for task in exp:
        
        stim_filenames = ["./stim/{}.txt".format(env_name) for env_name in TASKS[task]]
        room_filenames = ["./stim/{}.txt".format(env_name) for env_name in TASKS[task+"_room"]]
        stims = [readMap(stim_fn) for stim_fn in stim_filenames]
        room_keys = [isolate_rooms(readMap(stim_fn)) for stim_fn in room_filenames]

        # datas = query_specs(all_experiment_specs, lambda x: x['exp_name'] == "{}_los_real_{}".format(task, mode))
        file_to_read = open("1646155107.791908.pkl", "rb")
        datas = [pickle.load(file_to_read)]

        cmap =  cm.get_cmap("plasma").colors

        real_actions = datas[0]['actions']

        # Resim with los
        heatmap_agent = HeatmapAgent([datas[0]['states'][0]], real_actions, room_keys=[np.array(room) for room in room_keys], **exp_dict)
        real_states, observed_states, actions, rewards, run_infos  = heatmap_agent.run()

        # Heatmaps
        real_state = real_states[-1]
        observed_state = observed_states[-1]
        sy, sx = real_state.shape

        total = np.zeros((sy, sx))

        unobserved = (observed_state == np.ones(real_state.shape)*UNOBSERVED)
        wall = (real_state == np.ones(real_state.shape)*WALL)
        for syi in range(sy):
            for sxi in range(sx):
                if(not unobserved[syi, sxi]):
                    total[syi, sxi]+=1

        # More visible distinctions
        total = total
        total = (len(cmap)-1)*total/float(np.max(total))

        total_image = np.zeros((sy, sx, 3))
        for syi in range(sy):
            for sxi in range(sx):
                if(wall[syi, sxi]):
                    total_image[syi, sxi] = (0,0,0)
                else:
                    total_image[syi, sxi] = cmap[int(total[syi, sxi])]

        f = plt.figure(figsize=(32, 12))
        plt.imshow(total_image)
        plt.axis('off')
        plt.show()
        f.savefig("./notebooks/figs/{}.pdf".format(datas[0]['exp_name']), bbox_inches='tight')
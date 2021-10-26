            
# import pickle
from src.human_exp_database import HumanExperimentResults
from src.consts import *
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
from matplotlib import cm
import datetime
import math
from src.utils import *
from collections import defaultdict
matplotlib.use('TkAgg')

with open("./notebooks/human_exp1_results.pkl", 'rb') as handle:
    human_exps = pickle.load(handle)


cmap =  cm.get_cmap("plasma").colors
def get_agent(s):
    return (list(np.argwhere(s==AGENT_UP)) + list(np.argwhere(s==AGENT_DOWN)) + list(np.argwhere(s==AGENT_LEFT)) + list(np.argwhere(s==AGENT_RIGHT)))


human_cross_env_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))


LAZY=["8081841", "4625841", "7081743", "3049745", "0812297", "6272611", "0468704"]

load = False
similarity_results_filename = "./similarity_results.pkl"
c = []
if(load):
    with open(similarity_results_filename, 'rb') as handle:
        results = pickle.load(handle)
        points, c, human_cross_env_probs = results["points"], results["c"], results["human_cross_env_probs"]
else:
    # Heatmaps
    points = defaultdict(lambda: defaultdict(lambda:[]))
    modes = ["pomcp_simple", "pomcp_mle", "pomcp_ssp"]
    for tmapi, (tmap, exps) in enumerate(human_exps.datas.items()):
            for p1 in human_exps.datas[tmap].keys():
                if(p1 not in LAZY):
                    for mode in modes:
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
                            
                        # Run the simulation
                        print('{}_results_{}_{}.pkl'.format(mode, tmap, p1))
                        try:
                            results = read_from_minio('{}_results_{}_{}.pkl'.format(mode, tmap, p1), prefix="spatial_planning")
                            human_cross_env_probs[tmap][p1][mode] = results
                            print("Found:"+str(results))
                            # import sys
                            # sys.exit(1)
                        except:
                            print("Not found")

                    if(human_cross_env_probs[tmap][p1]["pomcp_simple"] is not None
                        and human_cross_env_probs[tmap][p1]["pomcp_mle"] is not None
                        and human_cross_env_probs[tmap][p1]["pomcp_ssp"] is not None):

                        v1 = [np.log(q) for q in human_cross_env_probs[tmap][p1]["pomcp_simple"] if q>0]
                        v2 = [np.log(q) for q in human_cross_env_probs[tmap][p1]["pomcp_mle"] if q>0]
                        v3 = [np.log(q) for q in human_cross_env_probs[tmap][p1]["pomcp_ssp"] if q>0]
                        v1m = np.sum(v1) if len(v1)>0 else 0.5
                        v2m = np.sum(v2) if len(v2)>0 else 0.5
                        v3m = np.sum(v3) if len(v3)>0 else 0.5
                        points["pomcp_simple"][p1].append(v1m)
                        points["pomcp_mle"][p1].append(v2m)
                        points["pomcp_ssp"][p1].append(v3m)
                        # c.append(tmapi)


agg_points_simple = {k: np.mean(v) for k, v in points["pomcp_simple"]}
agg_points_mle = {k: np.mean(v) for k, v in points["pomcp_mle"]}
agg_points_ssp = {k: np.mean(v) for k, v in points["pomcp_ssp"]}

sde_points_simple = {k: np.std(v)/np.sqrt(len(v)) for k, v in points["pomcp_simple"]}
sde_points_mle = {k: np.std(v)/np.sqrt(len(v)) for k, v in points["pomcp_mle"]}
sde_points_ssp = {k: np.std(v)/np.sqrt(len(v)) for k, v in points["pomcp_ssp"]}

minmin = min(min(min(agg_points_simple.values()), min(agg_points_mle.values())), min(agg_points_ssp.values()))

plt.scatter(points['pomcp_simple'], points['pomcp_mle'])
plt.title("Similarity Plot")
plt.xlabel("Uniform")
plt.ylabel("MLE")
plt.xlim([minmin, 0])
plt.ylim([minmin, 0])
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
plt.colorbar()
plt.figure()

# "pomcp_simple", "pomcp_ssp"
print(len(points['pomcp_simple']), len(points['pomcp_ssp']), len(c))
plt.scatter(points['pomcp_simple'], points['pomcp_ssp'])
plt.xlabel("Uniform")
plt.ylabel("Dist")
plt.xlim([minmin, 0])
plt.ylim([minmin, 0])
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
plt.colorbar()
plt.figure()

# "pomcp_mle", "pomcp_ssp"
plt.scatter(points['pomcp_mle'], points['pomcp_ssp'])
plt.xlabel("MLE")
plt.ylabel("Dist")
plt.xlim([minmin, 0])
plt.ylim([minmin, 0])
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
plt.colorbar()
plt.show()


import sys
sys.exit(1)



            
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
import matplotlib.cm as cm

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

shuffle = [2,5,0,3,4,1]

if(load):
    with open(similarity_results_filename, 'rb') as handle:
        results = pickle.load(handle)
        points, c, human_cross_env_probs = results["points"], results["c"], results["human_cross_env_probs"]
else:
    # Heatmaps
    points = defaultdict(list)
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

                    def sde(v):
                        return  np.std(v)/np.sqrt(len(v))

                    v1 = [q for q in human_cross_env_probs[tmap][p1]["pomcp_simple"] if q>0 and q<1]
                    v2 = [q for q in human_cross_env_probs[tmap][p1]["pomcp_mle"] if q>0 and q<1]
                    v3 = [q for q in human_cross_env_probs[tmap][p1]["pomcp_ssp"] if q>0]

                    v1m = np.mean(v1) if len(v1)>0 else 0.5
                    v2m = np.mean(v2) if len(v2)>0 else 0.5
                    v3m = np.mean(v3) if len(v3)>0 else 0.5

                    v1s = sde(v1)
                    v2s = sde(v2)
                    v3s = sde(v3)

                    points["pomcp_simple"].append(v1m)
                    points["pomcp_mle"].append(v2m)
                    points["pomcp_ssp"].append(v3m)

                    points["pomcp_simple_sde"].append(v1s)
                    points["pomcp_mle_sde"].append(v2s)
                    points["pomcp_ssp_sde"].append(v3s)
                    
                    c.append(shuffle[tmapi])



minmin = min(min(min(points['pomcp_simple']), min(points['pomcp_ssp'])), min(points['pomcp_mle']))
f = plt.figure()
sc = plt.scatter(points['pomcp_simple'], points['pomcp_mle'], c=c, alpha=0)

norm = matplotlib.colors.Normalize(vmin=min(c), vmax=max(c), clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap='plasma')
ce = [(mapper.to_rgba(ci)) for ci in c]


for i in range(len(points['pomcp_mle'])):
    plt.plot(points['pomcp_simple'][i], points['pomcp_mle'][i], c=ce[i], alpha=0.5)
    plt.errorbar(points['pomcp_simple'][i], points['pomcp_mle'][i], xerr = points['pomcp_simple_sde'][i], yerr = points['pomcp_mle_sde'][i], fmt="o", c=ce[i], alpha=0.5)


plt.title("Similarity Plot")
plt.xlabel("Uniform")
plt.ylabel("MLE")
plt.xlim([minmin, 1])
plt.ylim([minmin, 1])
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)

f.savefig("./notebooks/figs/simple_mle.pdf", bbox_inches='tight')
plt.show()

f = plt.figure()

sc = plt.scatter(points['pomcp_simple'], points['pomcp_ssp'], c=c, alpha=0)

for i in range(len(points['pomcp_simple'])):
    plt.plot(points['pomcp_simple'][i], points['pomcp_ssp'][i], c=ce[i], alpha=0.5)
    plt.errorbar(points['pomcp_simple'][i], points['pomcp_ssp'][i], xerr = points['pomcp_simple_sde'][i], yerr = points['pomcp_ssp_sde'][i], fmt="o", c=ce[i], alpha=0.5)


plt.xlabel("Uniform")
plt.ylabel("Dist")
plt.xlim([minmin, 1])
plt.ylim([minmin, 1])
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
f.savefig("./notebooks/figs/simple_ssp.pdf", bbox_inches='tight')
plt.show()

f = plt.figure()
sc = plt.scatter(points['pomcp_mle'], points['pomcp_ssp'], c=c, alpha=0)

for i in range(len(points['pomcp_mle'])):
    plt.plot(points['pomcp_mle'][i], points['pomcp_ssp'][i], c=ce[i], alpha=0.5)
    plt.errorbar(points['pomcp_mle'][i], points['pomcp_ssp'][i], xerr = points['pomcp_mle_sde'][i], yerr = points['pomcp_ssp_sde'][i], fmt="o", c=ce[i], alpha=0.5)

plt.xlabel("MLE")
plt.ylabel("Dist")
plt.xlim([minmin, 1])
plt.ylim([minmin, 1])
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
plt.show()

f.savefig("./notebooks/figs/mle_dist.pdf", bbox_inches='tight')




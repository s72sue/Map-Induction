            
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


cmap =  cm.get_cmap("plasma").colors
def get_agent(s):
    return (list(np.argwhere(s==AGENT_UP)) + list(np.argwhere(s==AGENT_DOWN)) + list(np.argwhere(s==AGENT_LEFT)) + list(np.argwhere(s==AGENT_RIGHT)))


human_cross_env_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))


load = False
exp="exp1"
similarity_results_filename = "./similarity_results.pkl"
c = []

exp1_levels = ["test1", "test2", "test3", "test4",  "test5",  "test6"]
exp2_levels = ["test7", "test8", "test9", "test10", "test11", "test12"]
exp1_participants = ["0812297", "2789611", "2900312", "2954467", "3049745", "3393196", "3844398", "4276108", "4625841", "5588827", "6272611", "6312290", "6385658", "6421078", "7081743", "7106784", "7305802", "7799324", "7862399", "7874247", "8023504", "8081841", "8126530", "8620459", "8667327", "9403949"]
exp2_participants = ['0350283', '0468704', '0686270', '1456053', '1653593', '2896074', '3393196', '3499382', '4020521', '4186111', '4435465', '4574413', '4832813', '4842726', '4887169', '5831456', '6028768', '6312290', '7374153', '7528739', '7543747', '8035917', '8108428', '8136373', '8349831', '8615341', '8760618', '9188145', '9366516', '9403949', '9479411', '9664366', '9722382', '9877970']

if(exp == "exp2"):
    levels = exp2_levels
    participants = exp2_participants
elif(exp == "exp1"):
    levels = exp1_levels
    participants = exp1_participants    

if(load):
    with open(similarity_results_filename, 'rb') as handle:
        results = pickle.load(handle)
        points, c, human_cross_env_probs = results["points"], results["c"], results["human_cross_env_probs"]
else:
    # Heatmaps
    points = defaultdict(list)
    modes = ["pomcp_simple", "pomcp_mle", "pomcp_ssp"]
    for tmapi, tmap in enumerate(levels):
        for p1 in participants:
            for mode in modes:
                print("Processing--{}--{}--{}".format(tmap, p1, mode))

                # Run the simulation
                print('{}_results_{}_{}.pkl'.format(mode, tmap, p1))
                try:
                    results = read_from_minio('{}_results_{}_{}.pkl'.format(mode, tmap, p1), prefix="spatial_planning")
                    human_cross_env_probs[tmap][p1][mode] = results
                    print("Found:"+str(results))
                except:
                    print("Not found")

            if(human_cross_env_probs[tmap][p1]["pomcp_simple"] is not None
                and human_cross_env_probs[tmap][p1]["pomcp_mle"] is not None
                and human_cross_env_probs[tmap][p1]["pomcp_ssp"] is not None):

                def sde(v):
                    return  np.std(v)/np.sqrt(len(v))

                v1 = [np.log(q) for q in human_cross_env_probs[tmap][p1]["pomcp_simple"] if q>0 and q<1]
                v2 = [np.log(q) for q in human_cross_env_probs[tmap][p1]["pomcp_mle"] if q>0 and q<1]
                v3 = [np.log(q) for q in human_cross_env_probs[tmap][p1]["pomcp_ssp"] if q>0 and q<1]

                v1m = np.mean(v1) if len(v1) > 0 else np.log(0.5)
                v2m = np.mean(v2) if len(v2) > 0 else np.log(0.5)
                v3m = np.mean(v3) if len(v3) > 0 else np.log(0.5)

                v1s = sde(v1)
                v2s = sde(v2)
                v3s = sde(v3)

                points["pomcp_simple"].append(v1m)
                points["pomcp_mle"].append(v2m)
                points["pomcp_ssp"].append(v3m)

                points["pomcp_simple_sde"].append(v1s)
                points["pomcp_mle_sde"].append(v2s)
                points["pomcp_ssp_sde"].append(v3s)
                
                c.append(tmapi)



print(points)
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
plt.xlim([minmin, 0])
plt.ylim([minmin, 0])
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)

f.savefig("./notebooks/figs/simple_mle_log.pdf", bbox_inches='tight')

f = plt.figure()

sc = plt.scatter(points['pomcp_simple'], points['pomcp_ssp'], c=c, alpha=0)

for i in range(len(points['pomcp_simple'])):
    plt.plot(points['pomcp_simple'][i], points['pomcp_ssp'][i], c=ce[i], alpha=0.5)
    plt.errorbar(points['pomcp_simple'][i], points['pomcp_ssp'][i], xerr = points['pomcp_simple_sde'][i], yerr = points['pomcp_ssp_sde'][i], fmt="o", c=ce[i], alpha=0.5)


plt.xlabel("Uniform")
plt.ylabel("Dist")
plt.xlim([minmin, 0])
plt.ylim([minmin, 0])
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
f.savefig("./notebooks/figs/simple_ssp_log.pdf", bbox_inches='tight')

f = plt.figure()
sc = plt.scatter(points['pomcp_mle'], points['pomcp_ssp'], c=c, alpha=0)

for i in range(len(points['pomcp_mle'])):
    plt.plot(points['pomcp_mle'][i], points['pomcp_ssp'][i], c=ce[i], alpha=0.5)
    plt.errorbar(points['pomcp_mle'][i], points['pomcp_ssp'][i], xerr = points['pomcp_mle_sde'][i], yerr = points['pomcp_ssp_sde'][i], fmt="o", c=ce[i], alpha=0.5)

plt.xlabel("MLE")
plt.ylabel("Dist")
plt.xlim([minmin, 0])
plt.ylim([minmin, 0])
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)

f.savefig("./notebooks/figs/mle_dist_log.pdf", bbox_inches='tight')




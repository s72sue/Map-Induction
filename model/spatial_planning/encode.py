import pickle5
import copy
from src.human_exp_database import HumanExperimentResults

with open("./notebooks/human_exp2_results_los.pkl", 'rb') as handle:
        human_exps = pickle5.load(handle)

# LAZY = ["8081841", "4625841", "7081743", "3049745", "0812297", "6272611"]
LAZY = ["8081841", "4625841", "7081743", "3049745", "0812297", "6272611"]

for human_id in human_exps.datas["test7"].keys():
        if(human_id not in LAZY):
                her = HumanExperimentResults()
                for m in human_exps.datas.keys():
                        her.datas[m] = {}
                        her.datas[m][human_id] = copy.deepcopy(human_exps.datas[m][human_id])
                with open("./notebooks/human_exp2_results_{}.pkl".format(human_id), 'wb') as nandle:
                        pickle5.dump(her, nandle)
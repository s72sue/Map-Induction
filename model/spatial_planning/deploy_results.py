import yaml
import time
import subprocess
import os
import pickle


assert os.environ['S3ACCESS_CUST']
assert os.environ['S3SECRET_CUST']

# Set up the hyperparameters here 
# for task in ["test1"]:
# 	for N in range(1):

with open("./notebooks/human_experiment_results.pkl", 'rb') as handle:
    human_exps = pickle.load(handle)

for mode in ["pomcp_simple", "pomcp_mle", "pomcp_ssp"]:
	for task in ["test1", "test1_Reflect", 
				 "test2", "test2_Reflect", 
				 "test3", "test3_Reflect", 
				 "test4", "test4_Reflect", 
				 "test5", "test5_Reflect", 
				 "test6", "test6_Reflect"]:
		for p1 in human_exps.datas[task].keys():
			with open("spatial_planning_results.yaml", 'r') as stream:
				try:
					conf = yaml.safe_load(stream)
				except yaml.YAMLError as exc:
					print(exc)

			setup_name = str(time.time())
			args_env = {"task_name": task,
						"exp_name": task+"_experiments_"+str(mode),
						"agent_name": mode,
						"search_depth": "20",
						"tree_queries": "500",
						"participant": "p"+p1,
						"discount_factor": "0.80",
						"observation_mode": "directional_line_of_sight",
						"num_iterations": "500",
						"save_dir": "./results/",
						"replan_strat": "every_step",
						"optimism": "0.0000001",
						"S3ACCESS_CUST":os.environ['S3ACCESS_CUST'],
						"S3SECRET_CUST":os.environ['S3SECRET_CUST']}

			converted_env = []
			for args_env_keys, args_env_values in args_env.items():
				converted_env.append({"name": args_env_keys, "value": args_env_values})

			conf['spec']['template']['spec']['containers'][0]['env'] = converted_env
			conf['metadata']['name'] = "spatial-planning-"+str(setup_name)

			conf_filename = './temp/tmp_delpoyment_conf_'+setup_name+'.yml'
			with open(conf_filename, 'w') as outfile:
			    yaml.dump(conf, outfile, default_flow_style=False)

			# Deploy
			print("Deploying "+str()+"...")
			subprocess.run(["kubectl", "apply", "-f", os.getcwd()+"/"+str(conf_filename)] )
			print("Deployed")

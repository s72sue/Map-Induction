#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 16
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=32000

#40

i=1
for mode in "pomcp_simple" "pomcp_mle" "pomcp_ssp"
do
	for task in "test7" "test7_Reflect" "test8" "test8_Reflect" "test9" "test9_Reflect" "test10" "test10_Reflect" "test11" "test11_Reflect" "test12" "test12_Reflect"
	do		
		if [ $((i)) -eq  $((SLURM_ARRAY_TASK_ID + 0)) ]; then
			python-jl evaluation.py --task-name="$task" --exp-name="$task"_exp2_runs1_"$mode" --agent-name="$mode" --search-depth="20" --tree-queries="100" --discount-factor="0.80" --observation_mode="directional_line_of_sight" --optimism="0.0000001" --num_iterations="500" --save-dir="./results/" --replan-strat="every_step" --write-external
		fi
		i=$((i+1))
	done
done

echo $i
echo ${SLURM_ARRAY_TASK_ID}
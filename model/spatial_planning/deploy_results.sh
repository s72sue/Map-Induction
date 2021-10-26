#!/bin/sh
#SBATCH --nodes=2
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8000

#1045

i=1
for mode in "pomcp_simple" "pomcp_mle" "pomcp_ssp"
do
	for task in "test1" "test1_Reflect" "test2" "test2_Reflect" "test3" "test3_Reflect" "test4" "test4_Reflect" "test5" "test5_Reflect" "test6" "test6_Reflect"
	do
		for p1 in '5588827' '8667327' '7799324' '8126530' '4276108' '2900312' '7862399' '8023504' '8081841' '6385658' '9403949' '3393196' '4625841' '6312290' '3049745' '2789611' '7081743' '0812297' '6272611' '3844398' '2954467' '7305802' '7106784' '4297577' '3842773' '7874247' '8620459' '3062964' '6421078'
		do
			if [ $((i)) -eq  $((SLURM_ARRAY_TASK_ID + 0)) ]; then
				python-jl notebooks/similarity_plots.py --task-name="$task" --exp-name="$task"_experiments_"$mode" --agent-name="$mode" --search-depth="20" --tree-queries="500" --discount-factor="0.80" --observation_mode="directional_line_of_sight" --optimism="0.0000001" --num_iterations="500" --participant=p"$p1" --save-dir="./results/" --replan-strat="every_step"
			fi
			i=$((i+1))
		done
	done
done

echo $i
echo ${SLURM_ARRAY_TASK_ID}
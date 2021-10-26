# spatial_planning

Install the required python and Julia librarires
```python -m pip install -r requirements.txt```
```julia packages.jl```


Run the evaluation file using python-jl with the desired arguments (explained below)
```python-jl evaluation.py```


## Arguments
Different experiments are run with argument flags.

### --task-name
The map sequence to use as the environment for the agent

### --agent-name
The name of the policy class to use in the environment. Below is a list of the available policies.
- *pomcp_simple:* Uniform model
- *pomcp_ssp:* Distributional model
- *pomcp_mle:* MAP Model
- *random_policy:* Random actions
- *landmark_policy:* Naive landmark-based policy
- *landmark_reward_policy:* Another naive landmark-based policy

### --search-depth
When using POMDP-based solvers, what is the maximum depth of the search tree.

### --tree-queries
When using POMDP-based solvers, how many queries should the solver make on a search tree for a single iteration.

### --num_iterations
Specifies the maximum number of steps in the environment

### --discount-factor
When using POMDP-based solvers, what is the assumed gamma in the POMDP model

### --optimism
Specifies the value of observing an unobserved cell

### --observation_mode
Specifies types of observations recieved by the agent

### --replan-strat
The replanning strategy to use at each step of execution.
- *mpc:* replan upon new observations
- *every_step:* replan every step

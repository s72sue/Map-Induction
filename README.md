# Map-Induction

Humans are expert explorers. Understanding computational cognitive mechanisms that support this efficiency can advance the study of the human mind, and enable more efficient exploration algorithms. 
Using a new behavioral Map Induction Task, we show that humans' exploration of new environments can be modeled as program induction -- humans  infer the structure of the unobserved spaces, based on priors over the distribution of observed spaces, and use this inference to optimize exploration. 
We show that this program induction process can be modeled computationally and quantitatively by a  Hierarchical Bayesian framework to improve exploration performance in AI.
We show that a modeling framework that learns inductive priors about spatial structures can outperform state of the art approximate Partially Observable Markov Decision Process planners, when applied to a realistic spatial navigation domain. 

## Datasets
The `datasets` folder contains anonymized datasets from the Map Induction Task (MIT), both from `Experiment1` and `Experiment2`. Both experiments contain the following two folders: <br><br>
`data`: Contains raw data in csv files collected during the experiment. Each csv file corresponds to a single subject. The csv files are named using randomly generated subject IDs with a unique ID per subject.<br><br>
`visual_results`: Contains visualizations of the the exploration trajectories of each subject. Each pdf file corresponds to a single subject and shows the order of stimulus presentation for the given subject. The pdf files are named using the subject ids. 


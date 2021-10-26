using Pkg
Pkg.update()
dependencies = [
    "PyCall",
    "IJulia",
    "Revise",
    "POMDPs",
    "POMDPModelTools",
    "POMDPPolicies",
    "POMDPSimulators",
    "Images",
    "Plots",
    "BasicPOMCP",
    "BeliefUpdaters",
    "ParticleFilters",
    "Reel",
    "Distributions",
    "POMDPLinter",
    "CPUTime",
    "Parameters",
    "Colors",
    "MCTS",
    "D3Trees"

]

Pkg.add(dependencies)
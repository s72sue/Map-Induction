import numpy as np
from src.matcher import obs_diff, obs_diff_novel, expand, get_unobs
from src.consts import *


# normalize so that the sum is 1
def normalize(dist):
    print(dist)
    return dist/np.sum(dist)

def uniform_prior():
    return 1

#size principle
def size_prior(proposal):
    return 1/(np.prod(proposal.shape))


def activation(x, power=5):
    return x**(1/power)

def get_reward_count(proposal):
    return np.count_nonzero(proposal==REWARD)

def get_reward_mismatch(proposal, observation):
    rewards_in_proposal = np.count_nonzero(proposal[observation == REWARD] != REWARD) == 0
    observed_proposal_rewards = np.count_nonzero(observation[(proposal == REWARD) * (observation != UNOBSERVED)] != REWARD) == 0
    return not rewards_in_proposal or not observed_proposal_rewards

# lik = fraction predicted * matched bits
def getBelief(hyps, map_space, obsmap=None, reward=False):
    belief = list()
    maps = list()
    prilst = list()
    liklst = list()
    for i in range(len(hyps)):
        h = hyps[i]
        C = np.count_nonzero(obs_diff(h, map_space))
        prior = uniform_prior()  #size_prior(submaps[i])
        if(get_reward_count(h)>0 and C==0):
            likelihood = 1
        else:
            likelihood = 0
       
        maps.append(h)
        prilst.append(prior)
        liklst.append(likelihood)

    if obsmap is not None:
        A = np.prod(obsmap.shape)
        n_unobs = len(get_unobs(obsmap))
        prilst.append(uniform_prior())  #size_prior(obsmap)
        liklst.append(1-n_unobs/A)
        maps.append(obsmap)
    prior = normalize(prilst)
    likelihood = normalize(liklst)
    belief = prior*likelihood
    belief = normalize(belief) 
    return belief, maps


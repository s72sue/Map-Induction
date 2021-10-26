import numpy as np
from src.utils import *
from src.programs import *
from src.consts import *
from collections import defaultdict
import itertools

# Discard tiling hypotheses that mismatch
# any of the observed space

def duplicatey(map_layout, k=2):
    "k: number of duplicates"
    (r,c) = map_layout.shape
    return np.resize(map_layout, (k*r,c))
 
def duplicatex(map_layout, k=2):
    "k: number of duplicates"
    (r,c) = map_layout.shape 
    return np.resize(map_layout, (r,k*c)) 


# expand a map given its
# topometric representation
def expand(prop, xy):
    """prop: ndarray that contains the proposal (metric)
       xy: (r,c) tuple specifying how the metric component 
           is connected to its copies (topological)
    """
    return np.tile(prop, (xy[0], xy[1]))
    
def visPH(prop, map_space):
    visualize(prop).show() 
    h, _ = map_hypothesis(prop, map_space)
    visualize(h).show()
    
    
# visualize all proposals and maps
def showPH(submapLib, hypLib, map_space):
    for (k,v), (k2,v2) in zip(submapLib.items(), hypLib.items()):
        for i in range(len(v)):
            print("Proposal/Map", i)
            visPH(v[i], map_space) 
        

def map_hypothesis(prop, map_space):
    map_rows, map_cols = map_space.shape
    prop_rows, prop_cols = prop.shape
    xy = (map_rows//prop_rows, map_cols//prop_cols)
    m = expand(prop, xy)
    return m, xy

def map_mult_hypothesis(props, map_space):
    map_rows, map_cols = map_space.shape
    prop_rows, prop_cols = props[0].shape
    xy = (map_rows//prop_rows, map_cols//prop_cols)
    total_m = []
    print("xy:"+str(xy))
    print(len(props))
    for order in itertools.product(*[list(range(len(props))) for _ in range(xy[0]*xy[1])]):
        m = np.zeros((map_rows, map_cols))
        o_idx = 0
        for i in range(xy[0]):
            for j in range(xy[1]):
                m[i*prop_rows:(i+1)*prop_rows, j*prop_cols:(j+1)*prop_cols] = props[order[o_idx]]
                o_idx+=1
        total_m.append(m)
    print("total m")
    print(len(total_m))
    return total_m, xy


def obs_diff(hypothesis, map_space):
    idxlst = (hypothesis!=UNOBSERVED)*(map_space!=UNOBSERVED)
    return hypothesis[idxlst] != map_space[idxlst]

def obs_diff_novel(hypothesis, map_space):
    r, c = map_space.shape
    diff = np.zeros((r,c))
    for ri in range(r):
        for ci in range(c):
            diff[ri, ci] = int(hypothesis[ri, ci] == UNOBSERVED)
           
    return diff


# rr: remove reward
def match(hypothesis, map_space, rr=True):
    assert hypothesis.shape == map_space.shape
    if (rr):
        hypothesis = remRewards(hypothesis)
        map_space = remRewards(map_space)

    diff = obs_diff(hypothesis, map_space)
    return np.count_nonzero(diff) == 0   


# returns list of indices of 
# unobserved cells in the map        
def get_unobs(map_space):
    result = np.where(map_space == UNOBSERVED)
    return list(zip(result[0], result[1]))


def exists(proplst, prop):
    flag = False
    zero = np.zeros_like(prop)
    diff = prop - proplst
    for x in range(len(diff)):
        if np.array_equal(diff[x],zero):
            flag = True   
    return flag

# s -> topometric representation of the 
# hypothesis map
def addtolib(key, submapLib, hypLib, p, s):
    if len(submapLib[key]) == 0 or not exists(submapLib[key], p):
        submapLib[key].append(p)
        hypLib[key].append(s)
    return submapLib, hypLib

def addtolib_multi(key, hypLib, h):
    hypLib[key].append(h)
    return hypLib

# s -> tuple that enables topometric representation
# Here matching is done without taking rewards into account, 
# just based on the spatial structure of the environment
def valid_multi_match(prop_dict, 
                      map_space, 
                      hypLib = defaultdict(list),
                      rr = True): 

    tophyps = list()
    print("Num prop dict items")
    print(prop_dict.keys())
    for key, value in prop_dict.items():
        if(len(value)>0):
            hs, s = map_mult_hypothesis(value, map_space) 
            for h in hs:
                m = match(h, map_space, rr=rr)
                if (m):  #match
                    addtolib_multi(key, hypLib, h)    
                    tophyps.append(h)

    return hypLib, tophyps

def valid_match(prop_dict, 
                map_space, 
                submapLib = defaultdict(list), 
                hypLib = defaultdict(list),
                rr = True): 

    submaps = list()
    tophyps = list()
    if submapLib is None:
        submapLib = dict()
        hypLib = dict()

    for key, value in prop_dict.items():

        for p in value:
            h, s = map_hypothesis(p, map_space) 
            #visualize(p).show()   #proposal
            #visualize(h).show()   #hypothesis 
            m = match(h, map_space, rr=rr)
            if (m):  #match
                addtolib(key, submapLib, hypLib, p, s)    
                submaps.append(p)
                tophyps.append(s)
    return submapLib, hypLib, submaps, tophyps


def count_submaps(submapLib):
    return sum([len(ls) for ls in submapLib.values() if isinstance(ls, list)])

# Do observations match an existing proposal or its reflection?  
# Here, rewards are taken into account when checking for a match
def match_existing(submapLib, hypLib, map_space, submaps=None, tophyps=None):
    submaps = list()
    tophyps = list()
    for (k,v), (k2,v2) in zip(submapLib.items(), hypLib.items()):
        for i in range(len(v)):
            hyp = expand(v[i], v2[i])
            m = match(hyp, map_space, rr=False)
            mRx = match(reflect(hyp, axis=0), map_space, rr=False)
            mRy = match(reflect(hyp, axis=1), map_space, rr=False)
            if (m):
                submaps.append(v[i])
                tophyps.append(v2[i])
            elif(mRx):
                submaps.append(reflect(v[i], axis=0)) # Is reflecting the entire hyp the same as reflecting a submap?
                tophyps.append(v2[i])
            elif(mRy):
                submaps.append(reflect(v[i], axis=1)) 
                tophyps.append(v2[i])    
    return submaps, tophyps

def match_existing_multi(hypLib, map_space, tophyps=None):
    tophyps = list()
    for (k2,hyps) in hypLib.items():
        for hyp in hyps:
            m = match(hyp, map_space, rr=False)
            tophyps.append(hyp)
    return tophyps


# replaces rewards with empty spaces
def remRewards(map_space):
    return np.where(map_space == 2, 0, map_space)
    

import numpy as np
from itertools import product
from src.consts import *
import copy
from src.utils import vis_single_grid, write_mapcode

# Proposal generator biases:
# 1. Each proposal should tile the maps space entirely
# 2. Set a lower limit on the size of the proposal


def get_percent_unobserved(grid):
    return float(np.count_nonzero(grid == UNOBSERVED))/(grid.shape[0]*grid.shape[1])

# returns all unique proposals from the proposal 
# space given a fixed proposal size
# unobserved cells are automatically discarded
def get_proposals(prop_space, xy):
    """prop_space: (ndarray) map from which to generate proposals
       xy: (tuple) size of proposals
    """
    rows, cols = prop_space.shape
    xdim, ydim = xy

    #list of unique proposals  
    #of dimensions xdim x ydim
    prop = []
    MAX_UNOBSERVED_RATIO = 0.75
    for i in range(0, rows-xdim+1, xdim):
        for j in range(0, cols-ydim+1, ydim):
            newp = prop_space[i:i+xdim,j:j+ydim]
            if get_percent_unobserved(newp) < MAX_UNOBSERVED_RATIO:
                ex=False
                for pi, p in enumerate(prop):
                    np = copy.copy(p)
                    np[newp == UNOBSERVED] = UNOBSERVED
                    nnewp = copy.copy(newp)
                    nnewp[p == UNOBSERVED] = UNOBSERVED
                    if((nnewp ==  np).all()):
                        prop[pi] = newp
                        ex=True
                if(not ex):
                    prop.append(newp)
                    if(len(prop)>2):
                        for pi, p in enumerate(prop):
                            write_mapcode("prop{}.txt".format(pi), p)
    return prop



# returns a sorted list of 
# factors of number n
def factors(n):    
    l1, l2 = [], []
    for i in range(1, int(n ** 0.5) + 1):
        q,r = n//i, n%i     # Alter: divmod() fn can be used.
        if r == 0:
            l1.append(i) 
            l2.append(q)    # q's obtained are decreasing.
    if l1[-1] == l2[-1]:    # To avoid duplication of the possible factor sqrt(n)
        l1.pop()
    l2.reverse()
    return l1 + l2
    


# max dim is map_space/2 for one of the dimensions
def gen_proposals(map_space, min_dim=(3,3), max_dim=(float("inf"),float("inf"))):
    r,c = map_space.shape
    fr = factors(r)
    fc = factors(c)
    xd, yd = min_dim
    xm, ym = max_dim
    
    # all combinations fr x fc
    # list of tuples [(1, 2), (1, 4)]
    comblst = list(product(fr, fc))

    # keys are tuples (1x2) and 
    #values are lists of proposals
    prop_dict = dict() 
    for i in range(len(comblst)):
        x,y = comblst[i]
        if x >= xd and y >= yd and x <= xm and y <= ym:
            prop_dict[comblst[i]] = get_proposals(map_space, comblst[i])       
    return prop_dict


# keys were area rather than dimensions    
# def gen_proposals(map_space, min_area=9):
#     r,c = map_space.shape
#     fr = factors(r)
#     fc = factors(c)
    
#     # all combinations fr x fc
#     # list of tuples [(1, 2), (1, 4)]
#     comblst = list(product(fr, fc))
#     arealst = [x * y for x, y in comblst]

#     # keys are tuples (1x2) and 
#     #values are lists of proposals
#     prop_dict = dict() 
#     for i in range(len(arealst)):
#         if arealst[i] >= min_area:
#             prop_dict[comblst[i]] = get_proposals(map_space, comblst[i])       
#     return prop_dict
    
    

# returns total number of proposals
def count_prop(prop_dict):
    s = 0
    for value in prop_dict.values():
        s += len(value)
    return s
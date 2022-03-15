
import os
import sys

if("PYTHONPATH" not in os.environ.keys()):
    sys.path.insert(0,'/om2/user/curtisa/spatial_planning')

import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import pickle
import csv
from src.utils import readMap, gen_gif, vis_single_grid, write_mapcode, \
    get_cardianals, in_state_bounds, num_rewards
import yaml
from collections import defaultdict
import numpy as np
from src.consts import *
import copy
import math
from src.human_exp_database import HumanExperimentResults
import julia

j = julia.Julia()
j.include('./src/pomdp/pomcp_planner_obs_rotational.jl')

# Import csv data
def load_human_experiment_pres(exp_filename, pres=1):
    retdict = {}
    key_set = []
    with open(exp_filename, 'r') as f_input:
        for iline, line in enumerate(f_input.readlines()):
            results_line = line.split("\n")[0].split(",")
            print(results_line[0])
            name = copy.deepcopy(results_line[0]).replace(" Reflect", "")
            if(pres == 0):
                if(" " not in name or name not in key_set):
                    retdict[results_line[0]] = results_line[1:]
            else:
                if(" " not in name or name in key_set):
                    retdict[results_line[0]] = results_line[1:]
            key_set.append(name)

    return retdict

def load_human_experiment(exp_filename):
    retdict = {}
    key_set = []
    with open(exp_filename, 'r') as f_input:
        for iline, line in enumerate(f_input.readlines()):
            results_line = line.split("\n")[0].split(",")
            print(results_line[0])
            name = copy.deepcopy(results_line[0])
            retdict[results_line[0]] = results_line[1:]
            key_set.append(name)

    return retdict
            
def get_human_data(results_folder="./human_experiments/exp1/"):
    # Get files within results folder
    results = []
    for f in listdir(results_folder):
        if isfile(join(results_folder, f)) and f[0] != '.':
            results.append(load_human_experiment(results_folder+f) )
    return results
    
    
def get_2d_midpoint(y1, x1, y2, x2):
    return (y1+y2)/2.0, (x1+x2)/2.0

def get_1d_midpoint(x1, x2):
    return (x1+x2)/2.0

def continuous_interpolation(ys, xs, drot):
    new_x = []
    new_y = []
    new_drots = []
    for i in range(xs.shape[0]-1):
        new_x.append(xs[i])
        new_y.append(ys[i])
        midy, midx = get_2d_midpoint(ys[i], xs[i], ys[i+1], xs[i+1])
        new_x.append(midx)
        new_y.append(midy)
        new_drots.append(drot[i])
        new_drots.append(drot[i])

    new_x.append(xs[-1])
    new_y.append(ys[-1])
    new_drots.append(drot[-1])

    return np.array(new_y), np.array(new_x), np.array(new_drots)
            

def map_shift(y, x, sy, sx, drots, tmap, setting_dir=""):
    settings = yaml.load(open(setting_dir+"/exp_params.yaml"), yaml.FullLoader)
    y = y + settings['alpha'] - settings[tmap]['yrem']
    point = np.array([y, x])
    grid_offset = np.array([ settings[tmap]['grid_offset_y'], settings[tmap]['grid_offset_x'] ])
    grid_spacing = np.array([ settings['grid_spacing_y'], settings['grid_spacing_x'] ])
    index = (point - grid_offset[:, None]) / grid_spacing[:, None] #  # np.round - rounds to nearest even value in case of *.5
   
    cy, cx = index
    cy = sy-cy
    return cy, cx, drots


def remove_redundancies(states, trajectory):
    new_states = []
    new_trajectory = []
    previous_state = None
    for state, step in zip(states, trajectory):
        if(previous_state is not None and np.count_nonzero(state - previous_state) != 0):
            new_states.append(state)
            new_trajectory.append(step)
        previous_state = state
    return new_states, new_trajectory


def get_inferred_actions(s1, s2):
    # All of this is assuming they can only move forward in the direction of the camera
    (y1, x1, absolute_rot), (y2, x2, d2) = s1, s2

    # First, infer the direction of the agent's next step (up, down, left, right)
    absolute_actions = []
    if(y2==y1+1 and x1==x2):
        absolute_actions.append(AGENT_DOWN)
    elif(y2==y1-1 and x1==x2):
        absolute_actions.append(AGENT_UP)
    elif(y2==y1 and x1==x2+1):
        absolute_actions.append(AGENT_LEFT)
    elif(y2==y1 and x1==x2-1):
        absolute_actions.append(AGENT_RIGHT)
    elif(y2==y1 and x1==x2):
        pass
    else:
        raise NotImplementedError

    relative_actions = [j.absolute_to_relative(absolute_action, absolute_rot) for absolute_action in absolute_actions]
    actions1 = []
    new_d = absolute_rot
    # See if rotating clockwise or counterclockwise is faster
    while(new_d != d2):
        new_d = j.get_rotation(copy.copy(new_d), "rotate_right")
        actions1.append("rotate_right")

    actions2 = []
    new_d = absolute_rot
    while(new_d != d2):
        new_d = j.get_rotation(copy.copy(new_d), "rotate_left")
        actions2.append("rotate_left")


    if(len(actions1)<len(actions2)):
        return (relative_actions+actions1)
    else:
        return (relative_actions+actions2)


def direction_from_camera_angle(camera_angle):
    clockwise_offset = 45
    ang = (camera_angle + clockwise_offset) % 360
    if( ang >= 0 and ang < 90):
        return AGENT_UP
    elif(ang >= 90  and ang < 180):
        return AGENT_RIGHT
    elif(ang >= 180 and ang < 270):
        return AGENT_DOWN
    elif(ang >= 270 and ang < 360):
        return AGENT_LEFT
    else:
        raise NotImplementedError



def fill_unreachable(state, ay, ax):
    ay = int(ay)
    ax = int(ax)
    new_state = np.ones(state.shape)*WALL
    start = (ay, ax)
    Q = [start]
    explored = [start]

    while(len(Q) > 0):

        q = Q.pop(0)
        y, x = q

        new_state[y, x] = state[y, x]

        for ny, nx in get_cardianals(y, x):
            neighbor = (ny, nx)
            if(neighbor not in explored and in_state_bounds(state, ny, nx) and not is_wall(state[ny,nx])):
                explored.append(neighbor)
                Q.append(neighbor)
            elif( in_state_bounds(state, ny, nx) and is_wall(state[ny,nx])):
                new_state[ny, nx] = state[ny, nx]

    return new_state


def bfs(state, 
        start_y, 
        start_x,
        start_d,
        is_goal = lambda state, y, x: True, 
        avoids = lambda state, y, x: False,
        max_bfs = 50):

    start = (start_y, start_x, start_d)
    Q = [start]
    explored = [start]
    parent = defaultdict(lambda: None)
    goal = None
    while(len(Q) > 0):
        if(len(explored) > max_bfs):
            return None

        q = Q.pop(0)
        y, x, d = q

        if(is_goal(state, y, x)):
            goal = q
            break
        
        for ny, nx in get_cardianals(y, x):
            neighbor = (ny, nx, start_d)
            if(neighbor not in explored and not avoids(state, ny, nx)):
                parent[neighbor] = q
                explored.append(neighbor)
                Q.append(neighbor)

    if(goal is None):
        return None

    # backtracking
    current = goal
    traj = [current]
    while(parent[current] is not None):
        current = parent[current]
        traj.append(current)

    return list(reversed(traj))

if __name__ == "__main__":


    # Keys
    XPOS = " X Position"
    ZPOS = " Z Position"
    DANG = " Camera Angle"
    Diamonds = " Diamonds"

    def get_grid_dist(ay, ax, ayc, axc):
        return math.sqrt((ay-ayc)**2+(ax-axc)**2)

    correct_wall_collisions = True # Currently the fix doesn't really work
    partially_observable = False

    exp="exp2"
    if(exp=="exp1"):
        MAPS_DIR = "./stim/exp1/"
        results_folder = "./human_experiments/exp1_data/"
    elif(exp=="exp2"):
        MAPS_DIR = "./stim/sim_exp3/"
        results_folder = "./human_experiments/exp2_data/"

    dicts = get_human_data(results_folder=results_folder)
    load=False
    if(load):
        with open('./notebooks/load_human_{}_results.pkl'.format(exp), 'rb') as handle:
            results = pickle.load(handle)
    else:
        results = HumanExperimentResults()
    
    her = HumanExperimentResults()
    for hdict in dicts:
        results = HumanExperimentResults()
        print("Participant: "+str(hdict["ID"]))
        for tmap_index in range(6):
            for r in [0]:
                norm = 1 if exp=="exp1" else 7

                if(r == 0):
                    tmap = "test{}".format(tmap_index+norm)
                    tmap_norm = "test{}".format(tmap_index+1)
                else:
                    tmap = "test{}_Reflect".format(tmap_index+norm)
                    tmap_norm = "test{}_Reflect".format(tmap_index+1)
        
                print("Working map: "+str(tmap))
                map_data = readMap("{}{}.txt".format(MAPS_DIR, tmap))
                sy, sx = map_data.shape
                print(hdict.keys())
                print(tmap_norm.replace("_", " ")+XPOS)
                assert (tmap_norm.replace("_", " ")+XPOS in hdict)
                if(tmap_norm.replace("_", " ")+XPOS in hdict):
                    xpos = np.array([float(p) for p in hdict[tmap_norm.replace("_", " ")+XPOS]])
                    zpos = np.array([float(p) for p in hdict[tmap_norm.replace("_", " ")+ZPOS]])
                    drot = np.array([float(p) for p in hdict[tmap_norm.replace("_", " ")+DANG]])
                    rewards = np.array([float(p) for p in hdict[tmap_norm.replace("_", " ")+Diamonds]])
                    print(rewards)
                    # num_rewards = len(num_rewards(map_data))
                    print("num_rewards: "+str(rewards[-1]))

                    print(len(num_rewards(map_data)))
                    # if(rewards[-1]  < len(num_rewards(map_data))):
                    #     print("Unfinished:"+str(hdict["ID"]))
                    #     print(rewards)
                    # else:
                    observation_mode = "room"
                    sy, sx = map_data.shape
                    zpos, xpos, drot = map_shift(zpos, xpos, sy, sx, drot, tmap, setting_dir = MAPS_DIR)

                    for i in range(zpos.shape[0]):
                        if(zpos[i] > 0 and xpos[i] > 0 and zpos[i] < sy and xpos[i] < sx):
                            map_data = fill_unreachable(map_data, zpos[i], xpos[i])
                            break


                    interpolation_factor = 4
                    for i in range(interpolation_factor):
                        cy, cx, ds = continuous_interpolation(zpos, xpos, drot)

                  
                    
                    fixing_trajectories=True
                    num_inter = 0
                        
                
                    trajectory = []
                    dcontdict = {}
                    for ii, i in enumerate(range(cy.shape[0])):
                        ay_cont, ax_cont, d_cont = cy[i], cx[i], ds[i]
                        if(ay_cont > 0 and ax_cont > 0 and ay_cont < sy and ax_cont < sx):
                            state = np.array(copy.deepcopy(map_data))
                            ay, ax, d = int(ay_cont), int(ax_cont), direction_from_camera_angle(d_cont)   
                            # First find the nearest cell to the discretized cell that isn't a wall
                            
                            
                            if(len(trajectory)>0):
                                res_states = None
                                unreachable = []
                                max_bfs = 1000
                                prev_ay, prev_ax, _ = trajectory[-1]
                                while(res_states is None):
                                  
                                    # print(state[ay-2: ay+2, ax-2: ax+2])
                                    (nearest_ay, nearest_ax, _) = bfs(state, ay, ax, d, 
                                                                      is_goal=lambda state, y, x: (state[y, x] == EMPTY and (y, x) not in unreachable),
                                                                      avoids=lambda state, y, x: (not in_state_bounds(state, y, x)),
                                                                      max_bfs=max_bfs)[-1]

                                    # Find a path from the previous square to this square and add to the trajectory
                                    res_states = bfs(state, prev_ay, prev_ax, d, 
                                                    is_goal=lambda state, y, x: (y==nearest_ay and x==nearest_ax), 
                                                    avoids=lambda state, y, x: (not in_state_bounds(state, y, x)) or is_wall(state[y, x]),
                                                    max_bfs=max_bfs)

                                    unreachable.append((nearest_ay, nearest_ax))
                                      
                                for res_state in res_states:           
                                    trajectory.append(res_state)
                                    dcontdict[res_state] = d_cont
                            else:
                                t = (ay, ax, d)
                                trajectory.append(t)
                                dcontdict[t] = d_cont


                          
                    # Final Discretization of trajectories
                    new_trajectory = []
                    for t in trajectory:
                        new_trajectory.append( (int(t[0]), int(t[1]), int(t[2])) )

                    trajectory = new_trajectory

                    current_observed_state = None
                    gif_states = []
                    real_states = []
                    actions = []
                    state = np.array(copy.deepcopy(map_data))
                    if(len(trajectory)==0):
                        print("Cannot process hdict")
                        break

                    state[trajectory[0][0], trajectory[0][1]] = trajectory[0][2]

                    # write_mapcode("./stim/sim_exp1/{}.txt".format(str(tmap)), state)

                    for t in range(len(trajectory)-1):
                        if(current_observed_state is None):
                            current_observed_state = j.init_observed_state(state,  observation_mode)
                            gif_states.append(current_observed_state)
                            real_states.append(state)
                        else:
                            for inferred_action in get_inferred_actions(trajectory[t], trajectory[t+1]):
                                state, current_observed_state, reward =  \
                                    j.next_state(copy.copy(state),
                                                 copy.copy(current_observed_state),
                                                 inferred_action,
                                                 observation_mode,
                                                 0.0)
                                gif_states.append(current_observed_state)
                                real_states.append(state)
                                actions.append(inferred_action)

                    gif_states, trajectory = remove_redundancies(gif_states, trajectory)
                    print("Adding result {} for map {}".format(hdict['ID'][0], tmap))
                    results.add_result(tmap, hdict['ID'][0], gif_states, real_states, actions, trajectory, hdict)

        
        with open("./notebooks/exp_results/human_experiment_results_{}.pkl".format(hdict["ID"][0]), 'wb') as handle:
            pickle.dump(results, handle)

    # with open('./notebooks/exp_results/human_{}_results_second.pkl'.format(exp), 'wb') as handle:
    #     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

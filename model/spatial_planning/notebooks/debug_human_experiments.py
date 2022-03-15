import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import pickle
import csv
from src.utils import readMap, gen_gif, vis_single_grid, write_mapcode, \
    get_cardianals, in_state_bounds
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
def load_human_experiment(exp_filename):
    retdict = {}
    with open(exp_filename, 'r') as f_input:
        for iline, line in enumerate(f_input.readlines()):
            results_line = line.split("\n")[0].split(",")
            retdict[results_line[0]] = results_line[1:]
    return retdict
            
def get_human_data(results_folder="./human_experiments/new_pilot_data/"):
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




    def get_grid_dist(ay, ax, ayc, axc):
        return math.sqrt((ay-ayc)**2+(ax-axc)**2)

    correct_wall_collisions = True # Currently the fix doesn't really work
    partially_observable = False

    exp="exp2"
    if(exp=="exp1"):
        MAPS_DIR = "./stim/exp1/"
        results_folder = "./human_experiments/new_pilot_data/"
    elif(exp=="exp2"):
        MAPS_DIR = "./stim/sim_exp3/"
        results_folder = "./human_experiments/exp2_data/"

    dicts = get_human_data(results_folder=results_folder)

    results = HumanExperimentResults()

    for tmap_index in [3]:
        for r in [0]:
            norm = 1 if exp=="exp1" else 7

            if(r == 0):
                tmap = "test{}".format(tmap_index+norm)
                tmap_norm = "test{}".format(tmap_index+1)
            else:
                tmap = "test{}_Reflect".format(tmap_index+norm)
                tmap_norm = "test{}_Reflect".format(tmap_index+1)

            print("Working map: "+str(tmap))
            for hdict in dicts:
                print("Participant: "+str(hdict["ID"]))
                # print(str(hdict["ID"][0])=="4832813")
                if(str(hdict["ID"][0])=="9722382"):
                    map_data = readMap("{}{}.txt".format(MAPS_DIR, tmap))

                    sy, sx = map_data.shape
                    plt.figure()
                    xpos = np.array([float(p) for p in hdict[tmap_norm.replace("_", " ")+XPOS]])
                    zpos = np.array([float(p) for p in hdict[tmap_norm.replace("_", " ")+ZPOS]])
                    drot = np.array([float(p) for p in hdict[tmap_norm.replace("_", " ")+DANG]])

                    observation_mode = "line_of_sight"
                    sy, sx = map_data.shape
                    zpos, xpos, drot = map_shift(zpos, xpos, sy, sx, drot, tmap, setting_dir = MAPS_DIR)

                    for i in range(zpos.shape[0]):
                        if(zpos[i] > 0 and xpos[i] > 0 and zpos[i] < sy and xpos[i] < sx):
                            map_data = fill_unreachable(map_data, zpos[i], xpos[i])
                            break

                    # im = vis_single_grid(map_data)
                    # im.save("./temp.png")
                    # import sys
                    # sys.exit(1)

                    interpolation_factor = 4
                    for i in range(interpolation_factor):
                        cy, cx, ds = continuous_interpolation(zpos, xpos, drot)

                    # print(ds)
                    # # for cyi in range(cy.shape[0]-1):
                    # #     print(int(cy[cyi]), int(cy[cyi+1]))

                    # import sys
                    # sys.exit()
                    
                    fixing_trajectories=True
                    num_inter = 0
                        
                
                    trajectory = []
                    gif_states = []
                    dcontdict = {}
                    for ii, i in enumerate(range(cy.shape[0])):
                        ay_cont, ax_cont, d_cont = cy[i], cx[i], ds[i]
                        if(ay_cont > 0 and ax_cont > 0 and ay_cont < sy and ax_cont < sx):
                            state = np.array(copy.deepcopy(map_data))
                            ay, ax, d = int(ay_cont), int(ax_cont), direction_from_camera_angle(d_cont)   
                            # First find the nearest cell to the discretized cell that isn't a wall
                            state[ay, ax] = AGENT
                            if(len(gif_states) == 0  or not np.array_equal(state, gif_states[-1])):
                                gif_states.append(state)
                                num_inter+=1
                        if(num_inter == 100):
                            break


                    gen_gif(gif_states)

                        
                        

                      
                
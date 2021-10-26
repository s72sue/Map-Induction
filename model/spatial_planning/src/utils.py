from src.map_grid import init_map_grid
import numpy as np
from minio import Minio
import os
import pickle
import shutil
from matplotlib import colors
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import time
from os.path import isfile, join
from src.consts import *
from os import listdir
import copy
from collections import defaultdict
import matplotlib; matplotlib.use('agg')


def readMap(file_name):
    file1 = open(file_name, 'r')
    lines = file1.readlines()
    fmap = []
    for ln, line in enumerate(lines):
        if(ln > 1):
            fmap_line = []
            for char in line.replace("\n", ""):
                fmap_line.append(ord(char)-ord('0'))
            fmap.append(fmap_line)
    return np.array(fmap)


def visualize(map_layout,
              start_pos=None,
              init_observations=None,
              block_size=20):
    mygrid = init_map_grid(map_layout, start_pos, init_observations, block_size)
    return mygrid


def vis_single_grid(grid):
    # create discrete colormap
    cmap = colors.ListedColormap(['black', 'white', 'yellow', 'red', 'blue', 'blue', "#f5f5f5", "green", "purple", "brown",
                                 '#969696', 'white', '#fdffc7', '#ffc7c7', '#c7cbff', '#c7cbff', "#f5f5f5", "#9cffa9", "#e2b0ff",  "#f5f5f5"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 
              9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(40, 48))

    ax.imshow(grid, cmap=cmap, norm=norm)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    im = Image.fromarray(image_from_plot)

    return im

# reads txt file contents as a
# string preserving the format
def read_mapcode(filename):
    f = open(filename, 'r+')
    file_data = f.read()
    f.close()
    return file_data


# writes the map array to txt file format
# used to store transformed maps
def write_mapcode(filename, data):
    # f = open(filename, 'w')
    # f.write(f"{data.shape[1]}\n")
    # f.write(f"{data.shape[0]}\n")
    # np.savetxt(f, data, fmt="%d", delimiter='', newline="\n")
    # f.close()

    text_file = open(filename, "w")
    data_str = [str(q)+"\n" for q in list(reversed(list(data.shape)))]
    for line in data:
        line_str = []
        for cell in line:
            line_str.append(chr(int(cell)+ord('0')))
        data_str.append("".join(line_str))
        data_str.append("\n")

    print(data_str)
    n = text_file.write("".join(data_str))
    text_file.close()



bucket_name = "spatial-planning"
def read_from_minio(file, prefix="spatial_planning"):
    # plot grasps in mayavi
    minioClient = Minio('s3.amazonaws.com',
                        access_key=os.environ["S3ACCESS_CUST"],
                        secret_key=os.environ["S3SECRET_CUST"],
                        secure=True)

    tmp_folder = "./tmp_minio/"
    if(not os.path.isdir(tmp_folder)):
        os.mkdir(tmp_folder)
    tmp_path = tmp_folder + file

    # s to get objects in folder
    if(not prefix in file):
        fn = prefix+"/"+file
    else:
        fn=file
    print(fn)
    minioClient.fget_object(bucket_name, fn , tmp_path)

    # Write data to a pickle file
    with open(tmp_path, 'rb') as handle:
        b = pickle.load(handle)
    return b


def write_to_minio(obj, name):
    print("Saving minio: "+str(name))
    # plot grasps in mayavi
    minioClient = Minio('s3.amazonaws.com',
                        access_key=os.environ["S3ACCESS_CUST"],
                        secret_key=os.environ["S3SECRET_CUST"],
                        secure=True)

    tmp_folder = "./tmp_minio/"
    if(not os.path.isdir(tmp_folder)):
        os.mkdir(tmp_folder)
    tmp_filename = str(name)+".pkl"
    tmp_path = tmp_folder + tmp_filename
    # Write data to a pickle file
    with open(str(tmp_path), 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Write file to bucket
    minioClient.fput_object(bucket_name, "spatial_planning/"
                            + str(tmp_filename), str(tmp_path))
    shutil.rmtree(tmp_folder)

# Visualization
def gen_gif(grids, hyps=None, temp_dir = "../temp/"):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    images = []
    filenames = []
    for action_idx, grid in enumerate(grids):
        if( not (hyps is None) ):
            gy, gx = grid.shape
            old_grid = np.ones((gy, gx))*UNOBSERVED
            old_grid[0:gy, :] = copy.copy(grid)
            if(len(hyps[action_idx])>0):
                old_grid[0:gy, :] = copy.copy(grid)
                mask = np.where(grid == UNOBSERVED)
                for y, x in zip(mask[0], mask[1]):
                    old_grid[y, x] = hyps[action_idx][0][y, x] + 10

            grid = old_grid
        im = vis_single_grid(grid)
        im.save(temp_dir + str(action_idx) + ".png")
        filenames.append(temp_dir + str(action_idx) + ".png")
    
    print("Reading Images..")
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
        
    print("Saving...")
    imageio.mimsave(temp_dir + str(time.time()) + ".gif", images)

# Loading local
def load_experiment(exp_filename):
    file_to_read = open(exp_filename, "rb")
    
    loaded_dictionary = pickle.load(file_to_read)
    return loaded_dictionary

def load_all_experiments(results_dir):
    return [load_experiment(results_dir+f) for f in listdir(results_dir) if isfile(join(results_dir, f)) and f[0] != '.']

def load_all_exp_name(exp_name, results_dir = "../results/"):
    return [exp for exp in load_all_experiments(results_dir) if "exp_name" in exp.keys() and exp["exp_name"] == exp_name]


def query_specs(specs, query):
    minioClient = Minio('s3.amazonaws.com',
                      access_key=os.environ["S3ACCESS_CUST"],
                      secret_key=os.environ["S3SECRET_CUST"],
                      secure=True)
    total_datas = []
    for name, spec in specs.items():
        if(query(spec)):
            # Get name from spec
            print("Loading data: "+str(name))
            total_datas.append(read_from_minio(name))
    return total_datas


def load_all_experiment_specs_minio(prefix="spatial_planning/"):
    
    minioClient = Minio('s3.amazonaws.com',
                      access_key=os.environ["S3ACCESS_CUST"],
                      secret_key=os.environ["S3SECRET_CUST"],
                      secure=True)

    exps = {}
    for obj in minioClient.list_objects(bucket_name, prefix = prefix):
        if(obj.object_name.split("/")[-1].startswith("spec_")):
            print("Checking spec: "+str(obj.object_name))
            exps[obj.object_name.replace("spec_", "")] = read_from_minio(obj.object_name)
            
    return exps

def get_cardianals(y, x):
    return [
        (y-1, x+0),
        (y+1, x+0),
        (y+0, x+1),
        (y+0, x-1)
    ]

def in_state_bounds(state, y, x):
    sy, sx = state.shape
    return y>=0 and x>=0 and y<sy and x<sx


def num_rewards(state_map):
    reward_squares = []
    for y in range(state_map.shape[0]):
        for x in range(state_map.shape[1]):
            if(state_map[y, x] == REWARD):
                reward_squares.append((y, x))
    return reward_squares


def find_reward(state_map, observation):
    reward_squares = []
    for y in range(state_map.shape[0]):
        for x in range(state_map.shape[1]):
            if(state_map[y, x] == REWARD and (observation[y, x] == REWARD or observation[y, x] == UNOBSERVED)):
                reward_squares.append((y, x))

    return reward_squares


def find_agent(observation):
    for y in range(observation.shape[0]):
        for x in range(observation.shape[1]):
            if(is_agent(observation[y, x])):
                return y, x
    assert False, "Agent not on grid"

def find_reachable_unobserved(observation):
    # TODO: BFS from current position
    ay, ax = find_agent(observation)
    Q = [(ay, ax)]
    explored = defaultdict(lambda: False)
    while(len(Q) > 0):
        q = Q.pop(0)
        y, x = q
        if(observation[y, x] == UNOBSERVED):
            return y, x

        for ny, nx in get_cardianals(y, x):
            neighbor = (ny, nx)
            if(not explored[neighbor] and in_state_bounds(observation, ny, nx) and (observation[ny, nx] == UNOBSERVED or observation[ny, nx] == EMPTY) ) :
                explored[neighbor] = True
                Q.append(neighbor)

    assert False, "Everything is observed"

# def fill_obsmap(observation):
#     new_state = np.zeros(observation.shape)
#     oy, ox = find_reachable_unobserved(observation)
#     start = (oy, ox, 0)
#     Q = [start]
#     explored = defaultdict(lambda: False)
#     while(len(Q) > 0):
#         q = Q.pop(0)
#         y, x, depth = q
#         rval = 1.0/float((depth)+1)
#         if(rval > new_state[y, x]):
#             new_state[y, x] = rval

#         for ny, nx in get_cardianals(y, x):
#             neighbor = (ny, nx, depth+1)
#             if(not explored[(neighbor[0], neighbor[1])] and in_state_bounds(new_state, ny, nx) and (observation[ny, nx] == EMPTY or observation[ny, nx] == UNOBSERVED)) :
#                 explored[(neighbor[0], neighbor[1])] = True
#                 Q.append(neighbor)
#     return new_state, (oy, ox)

def fill_obsmap(observation):
    new_state = np.zeros(observation.shape)
    Q = []
    explored = defaultdict(lambda: False)
    for y in range(observation.shape[0]):
        for x in range(observation.shape[1]):
            if(observation[y, x] == UNOBSERVED):
                Q.append((y, x, 0))
                explored[(y, x)]=True
                new_state[y, x]=1.0

    frontier = [(y, x) for (y, x, d) in Q]

    while(len(Q) > 0):
        q = Q.pop(0)
        (y, x, depth) = q
        rval = 1.0/float((depth)+1)
        if(rval > new_state[y, x]):
            new_state[y, x] = rval

        for ny, nx in get_cardianals(y, x):
            neighbor = (ny, nx, depth+1)
            if(not explored[(neighbor[0], neighbor[1])] and in_state_bounds(new_state, ny, nx) and (observation[ny, nx] == EMPTY or observation[ny, nx] == UNOBSERVED)):
                explored[(neighbor[0], neighbor[1])] = True
                Q.append(neighbor)
    return new_state, frontier

def fill_reward(state_map, observation):
    large_dim = sum(state_map.shape)
    new_state = np.zeros(state_map.shape)
    rlocs = find_reward(state_map, observation)
    filled_states = []
    for rloc in rlocs:
        new_state = np.zeros(state_map.shape)
        ry, rx = rloc
        start = (ry, rx, 0)
        Q = [start]
        explored = defaultdict(lambda: False)
        while(len(Q) > 0):
            q = Q.pop(0)
            y, x, depth = q
            rval = 1.0/float((depth)+1)
            if(rval > new_state[y, x]):
                new_state[y, x] = rval

            for ny, nx in get_cardianals(y, x):
                neighbor = (ny, nx, depth+1)
                if(not explored[(neighbor[0], neighbor[1])] and in_state_bounds(new_state, ny, nx) and (state_map[ny, nx] == EMPTY or state_map[ny, nx] == UNOBSERVED)) :
                    explored[(neighbor[0], neighbor[1])] = True
                    Q.append(neighbor)
        filled_states.append(new_state)
    return filled_states
    

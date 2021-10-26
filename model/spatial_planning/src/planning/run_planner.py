import os
import random
import sys
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from src.planning.planner import find_plan, apply_action, task_from_domain_problem, \
    t_get_action_instances, parse_sequential_domain, parse_problem, plan, parse_solution, read
from src.planning.problem_generator import Generator
from PIL import Image
from matplotlib import colors
from src.utils import *
from src.consts import semantics_map, inverse_semantics_map
import copy 

matplotlib.use('TkAgg')

class GridProblemGen(Generator):

    def exp_from_filename(self, filename):
        return filename.split("/")[-1].split(".txt")[0]

    def __init__(self, grid_filename="./stim/chain.txt", TMPFILE = "./src/planning/temp", start_position = (9, 1) ):
        # TODO: Encode start_position in the grid file

        if not os.path.exists(TMPFILE):
            os.makedirs(TMPFILE)

        self.grid_filename = grid_filename
        self.start_position = start_position
        self.grid = readMap(grid_filename)
        self.problem_name = "Navigation_" + self.exp_from_filename(self.grid_filename )
        self.domain_name = "Navigation"
        self.problem_file_path = os.path.join(TMPFILE, str(self.problem_name) + ".pddl")

    def get_objects(self):
        objects = []
        for pi in range(self.grid.shape[0]):
            for pj in range(self.grid.shape[1]):
                objects.append("pos-%s-%s" % (pi, pj))
        return objects

    def get_init(self):
        init = []
        total_set = []
        for pi in range(self.grid.shape[0]):
            for pj in range(self.grid.shape[1]):
                if (pi - 1 >= 0):
                    init.append(['adjacent', "pos-%s-%s" % (pi, pj), "pos-%s-%s" % (pi - 1, pj)])
                if (pj - 1 >= 0):
                    init.append(['adjacent', "pos-%s-%s" % (pi, pj), "pos-%s-%s" % (pi, pj - 1)])
                if (pi + 1 < self.grid.shape[0]):
                    init.append(['adjacent', "pos-%s-%s" % (pi, pj), "pos-%s-%s" % (pi + 1, pj)])
                if (pj + 1 < self.grid.shape[1]):
                    init.append(['adjacent', "pos-%s-%s" % (pi, pj), "pos-%s-%s" % (pi, pj + 1)])
                total_set.append("pos-%s-%s" % (pi, pj))

        # Use the seed to create a distribution of boundaries and rewards
        for pi in range(self.grid.shape[0]):
            for pj in range(self.grid.shape[1]):
                init.append([semantics_map[self.grid[pi][pj]], "pos-%s-%s" % (pi, pj)])

        init.append(["agent", "pos-%s-%s" % self.start_position]);
        return init

    @staticmethod
    def get_grid_state(grid, state):

        grid = np.zeros((grid.shape[0], grid.shape[1]))
   
        for atom in state:
            i, j = int(str(atom.args[0]).split("-")[1]), int(str(atom.args[0]).split("-")[2])
            
            if(atom.predicate in inverse_semantics_map and inverse_semantics_map[atom.predicate]>grid[i][j]):
                grid[i][j] = inverse_semantics_map[atom.predicate]
        
        return grid

    def get_goal(self):
        goal = []
        for pi in range(self.grid.shape[0]):
            for pj in range(self.grid.shape[1]):
                goal.append(['not', 'reward', "pos-%s-%s" % (pi, pj)])
        return goal


def vis_grid(grid):
    # create discrete colormap
    cmap = colors.ListedColormap(['black', 'green', 'yellow', 'red', 'blue', 'blue'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap=cmap, norm=norm)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)


    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    im = Image.fromarray(image_from_plot)

    return im


if __name__ == "__main__":


    # Generate problem file
    gp = GridProblemGen()
    gp.generate()

    # Read in domain and problem files
    domain_file = "./src/planning/grid_domain.pddl"
    plan_filename = plan(domain_file,gp.problem_file_path)
    plan = parse_solution(plan_filename)
    print(plan)

    # Get the states
    transitions = []
    task_domain = parse_sequential_domain(read(domain_file))
    task_problem = parse_problem(task_domain, read(gp.problem_file_path))

    # Get the initial state of the problem and step through
    task = task_from_domain_problem(task_domain, task_problem)
    action_instances = t_get_action_instances(task, plan)

    concrete_states = [task.init]
    grid = copy.deepcopy(GridProblemGen.get_grid_state(gp.grid, task.init))
    im = vis_grid(grid)
    filename_gen = lambda a, ext: "./src/planning/temp/a" + str(a) + "." + ext
    filenames = [filename_gen(0, "png")]
    im.save(filename_gen(0, "png"))

    for action_idx, action in enumerate(action_instances):
        state = apply_action(concrete_states[-1], action)
        concrete_states.append(state)
        grid = copy.deepcopy(GridProblemGen.get_grid_state(grid, state))
        im = vis_grid(grid)
        im.save(filename_gen(action_idx, "png"))
        filenames.append(filename_gen(action_idx, "png"))


    # Turn the images into a gif
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(filename_gen(-1, "gif"), images)

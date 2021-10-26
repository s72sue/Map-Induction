from src.map_grid import init_map_grid
import numpy as np


# string name can be a string, list
# or an arrapy (rows will be reversed)
def reverse_str(stringname):
    return stringname[::-1] 

def reflect(map_layout, axis=1):
    """
     axis=0: reflect vertically up/down
     axis=1: reflect horizontally left/right
    """
    return np.flip(map_layout, axis=axis)

def remove_wall(map_layout, indx):
    "indx: tuple (i,j)"
    layout = np.copy(map_layout)
    layout[indx] = 0
    return layout

def add_wall(map_layout, indx):
    "indx: tuple (i,j)"
    layout = np.copy(map_layout)
    layout[indx] = 3
    return layout

def rotate(map_layout, k=1):
    "k: # of times to rotate counterclockwise by 90deg"
    return np.rot90(map_layout, k=k, axes=(0, 1))

def duplicatey(map_layout, k=2):
    "k: number of duplicates"
    (r,c) = map_layout.shape
    return np.resize(map_layout, (k*r,c))
 
def duplicatex(map_layout, k=2):
    "k: number of duplicates"
    (r,c) = map_layout.shape
    layout = np.resize(map_layout.T, (k*c,r))  
    return rotate(layout, k=3)


# uniform scaling in x and y directions
def scaleu(map_layout, k=2):
    "k: amount to scale by"
    return np.repeat(np.repeat(map_layout, k, axis=0), k, axis=1)


# scaleds in x dir by repeating columns
def scalex(map_layout, k=2):
    "k: amount to scale by"
    return np.repeat(map_layout, k, axis=1)


# scales in y dir by repeating rows
def scaley(map_layout, k=2):
    "k: amount to scale by"
    return np.repeat(map_layout, k, axis=0)
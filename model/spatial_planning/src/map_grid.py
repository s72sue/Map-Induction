from ipythonblocks import ImageGrid

def init_map_grid(map, start_pos = None, init_observations = None, block_size=20):
    grid_columns = map.shape[1]
    grid_rows = map.shape[0]
    grid = ImageGrid(grid_columns, grid_rows, block_size=block_size, origin='upper-left')
        
    for row in range(grid_rows):
        for column in range(grid_columns):
            # grey - unobserved
            if map[row,column] == 6:
                grid[column,row] = (128, 128, 128)
            # black - empty space
            if map[row,column] == 0:
                grid[column,row] = (0, 0, 0)
            # red - wall
            if map[row,column] == 3:
                grid[column,row] = (153, 0, 0)
            # blue - start pos
            if map[row,column] == 5:
                grid[column,row] = (0, 0, 255)
            # reward locations
            if map[row,column] == 2:
                grid[column,row] = (255,255,0) 

    if start_pos is not None:
        x_pos = start_pos[0]
        y_pos = start_pos[1]
        grid[x_pos, y_pos] = (0, 0, 255)

    if init_observations is not None:
        makeObserved(grid, init_observations)
            
    return grid


# changes the grid to mark list of observations
# as observed. Observations=[(3,2),(2,2)]
def makeObserved(grid, observations):
    for position in observations:
        grid[position[0],position[1]] = (255, 255, 255)


# save the grid as an image (extension=pdf/png)
def makeImage(grid, node_name, extension='pdf'):
    grid.save_image(node_name + "." + extension)            


# changes the grid to mark new observations  
# and current location. 
def updateGrid(grid, agent_location, observations):
    for block in grid:
        if block.rgb == (128, 128, 128):
            block.rgb = (255, 255, 255)
    
    grid[agent_location[0], agent_location[1]] = (0, 0, 255)
    
    makeObserved(grid, observations)
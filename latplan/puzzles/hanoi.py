
import numpy as np
from .model.hanoi import generate_configs, successors, config_state

## patterns ##############################################################

patterns = [
    [[0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],],
    [[1,1,0,0,1,1,0,0,],
     [1,1,0,0,1,1,0,0,],
     [0,0,1,1,0,0,1,1,],
     [0,0,1,1,0,0,1,1,],
     [1,1,0,0,1,1,0,0,],
     [1,1,0,0,1,1,0,0,],
     [0,0,1,1,0,0,1,1,],
     [0,0,1,1,0,0,1,1,],],
    [[1,0,0,0,1,0,0,0,],
     [0,1,0,0,0,1,0,0,],
     [0,0,1,0,0,0,1,0,],
     [0,0,0,1,0,0,0,1,],
     [1,0,0,0,1,0,0,0,],
     [0,1,0,0,0,1,0,0,],
     [0,0,1,0,0,0,1,0,],
     [0,0,0,1,0,0,0,1,],],
    [[0,0,0,0,0,0,0,0,],
     [0,1,1,1,0,0,0,0,],
     [0,1,1,1,0,0,0,0,],
     [0,1,1,1,0,0,0,0,],
     [0,0,0,0,1,1,1,0,],
     [0,0,0,0,1,1,1,0,],
     [0,0,0,0,1,1,1,0,],
     [0,0,0,0,0,0,0,0,],],
    [[0,0,0,1,0,0,0,1,],
     [0,0,1,0,0,0,1,0,],
     [0,1,0,0,0,1,0,0,],
     [1,0,0,0,1,0,0,0,],
     [0,0,0,1,0,0,0,1,],
     [0,0,1,0,0,0,1,0,],
     [0,1,0,0,0,1,0,0,],
     [1,0,0,0,1,0,0,0,],],
    [[0,0,0,1,1,0,0,0,],
     [0,0,1,0,0,1,0,0,],
     [0,1,0,0,0,0,1,0,],
     [1,0,0,0,0,0,0,1,],
     [0,0,0,1,1,0,0,0,],
     [0,0,1,0,0,1,0,0,],
     [0,1,0,0,0,0,1,0,],
     [1,0,0,0,0,0,0,1,],],
    [[1,0,0,0,0,0,0,1,],
     [0,1,0,0,0,0,1,0,],
     [0,0,1,0,0,1,0,0,],
     [0,0,0,1,1,0,0,0,],
     [1,0,0,0,0,0,0,1,],
     [0,1,0,0,0,0,1,0,],
     [0,0,1,0,0,1,0,0,],
     [0,0,0,1,1,0,0,0,],],
    [[0,0,0,0,0,0,0,0,],
     [0,0,0,0,1,1,1,0,],
     [0,0,0,0,1,0,1,0,],
     [0,0,0,0,1,1,1,0,],
     [0,1,1,1,0,0,0,0,],
     [0,1,0,1,0,0,0,0,],
     [0,1,1,1,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],],
    [[1,0,0,0,1,0,0,0,],
     [1,0,0,0,1,0,0,0,],
     [0,1,0,0,0,1,0,0,],
     [0,1,0,0,0,1,0,0,],
     [0,0,1,0,0,0,1,0,],
     [0,0,1,0,0,0,1,0,],
     [0,0,0,1,0,0,0,1,],
     [0,0,0,1,0,0,0,1,],],
    [[0,0,0,1,0,0,0,1,],
     [0,0,0,1,0,0,0,1,],
     [0,0,1,0,0,0,1,0,],
     [0,0,1,0,0,0,1,0,],
     [0,1,0,0,0,1,0,0,],
     [0,1,0,0,0,1,0,0,],
     [1,0,0,0,1,0,0,0,],
     [1,0,0,0,1,0,0,0,],],
    [[1,0,1,0,1,0,1,0,],
     [1,0,1,0,1,0,1,0,],
     [1,0,1,0,1,0,1,0,],
     [1,0,1,0,1,0,1,0,],
     [1,0,1,0,1,0,1,0,],
     [1,0,1,0,1,0,1,0,],
     [1,0,1,0,1,0,1,0,],
     [1,0,1,0,1,0,1,0,],],
    [[1,1,1,1,1,1,1,1,],
     [0,0,0,0,0,0,0,0,],
     [1,1,1,1,1,1,1,1,],
     [0,0,0,0,0,0,0,0,],
     [1,1,1,1,1,1,1,1,],
     [0,0,0,0,0,0,0,0,],
     [1,1,1,1,1,1,1,1,],
     [0,0,0,0,0,0,0,0,],],
    [[0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],
     [1,1,1,1,1,1,1,1,],
     [1,1,1,1,1,1,1,1,],
     [1,1,1,1,1,1,1,1,],
     [1,1,1,1,1,1,1,1,],
     [0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],],
    [[0,0,1,1,1,1,0,0,],
     [0,0,1,1,1,1,0,0,],
     [0,0,1,1,1,1,0,0,],
     [0,0,1,1,1,1,0,0,],
     [0,0,1,1,1,1,0,0,],
     [0,0,1,1,1,1,0,0,],
     [0,0,1,1,1,1,0,0,],
     [0,0,1,1,1,1,0,0,],],
]

patterns = np.array(patterns)

## code ##############################################################

def generate1(config,disks,towers):
    l = len(config)
    disk_height = 8
    disk_inc = disk_height // 2
    base_disk_width_factor = 2
    base_disk_width = disk_height * base_disk_width_factor

    border = 1
    tower_width  = disks * (2*disk_inc) + base_disk_width + border
    tower_height = disks*disk_height
    
    figure = np.ones([tower_height,
                      tower_width*towers+2],dtype=np.int8)
    state = config_state(config,disks,towers)
    # print(l,figure.shape)
    # print(config)
    # print(state)
    for i, tower in enumerate(state):
        tower.reverse()
        # print(i,tower)
        x_center = tower_width *  i + disks * disk_inc # lacks base_disk_width
        
        x_left  = tower_width *  i
        x_right = tower_width * (i+1) - border
        for j,disk in enumerate(tower):
            # print(j,disk,(l-j)*2)
            figure[
                tower_height - disk_height * (j+1) :
                tower_height - disk_height * j,
                x_center - disk * disk_inc :
                x_center + disk * disk_inc + base_disk_width] \
                = 0
                # = np.tile(patterns[disk],(1,disk+base_disk_width_factor))
    return figure

def generate(configs,disks,towers):
    return np.array([ generate1(c,disks,towers) for c in configs ])
                

def states(disks, towers, configs=None):
    if configs is None:
        configs = generate_configs(disks, towers)
    return generate(configs,disks,towers)

def transitions(disks, towers, configs=None, one_per_state=False):
    if configs is None:
        configs = generate_configs(disks, towers)
    if one_per_state:
        def pickone(thing):
            index = np.random.randint(0,len(thing))
            return thing[index]
        transitions = np.array([
            generate([c1,pickone(successors(c1,disks,towers))],disks,towers)
            for c1 in configs ])
    else:
        transitions = np.array([ generate([c1,c2],disks,towers)
                                 for c1 in configs for c2 in successors(c1,disks,towers) ])
    return np.einsum('ab...->ba...',transitions)



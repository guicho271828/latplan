import numpy as np


panels = [
    [[0, 0, 0, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 0, 1, 0, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 0, 1, 0, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 0, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 0, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 0, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 0, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 0, 0, 0,],
     [0, 0, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 0, 0, 0,],
     [0, 1, 0, 0, 0,],
     [0, 1, 1, 0, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 0, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 0, 0, 0,],
     [0, 0, 1, 1, 0,],
     [0, 1, 0, 0, 0,],
     [0, 1, 0, 0, 0,],
     [0, 0, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 0, 0, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 0, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 0, 0,],
     [0, 0, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 1, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 1, 0, 0,],
     [0, 0, 1, 0, 0,],],
]

def generate_configs(digit=9):
    import itertools
    return itertools.permutations(range(digit))

def generate_puzzle(configs, width, height):
    assert width*height <= 16
    base_width = 5
    base_height = 6
    dim_x = base_width*width
    dim_y = base_height*height
    def generate(config):
        figure = np.zeros((dim_y,dim_x))
        for digit,pos in enumerate(config):
            x = pos % width
            y = pos // width
            figure[y*base_height:(y+1)*base_height,
                   x*base_width:(x+1)*base_width] = panels[digit]
        return figure
    return np.array([ generate(c) for c in configs ]).reshape((-1,dim_y,dim_x))

def successors(config,width,height):
    pos = config[0]
    x = pos % width
    y = pos // width
    print x,y
    succ = []
    if x is not 0:
        c = list(config)
        other = next(i for i,_pos in enumerate(c) if _pos == pos-1)
        c[0] -= 1
        c[other] += 1
        succ.append(c)
    if x is not width-1:
        c = list(config)
        other = next(i for i,_pos in enumerate(c) if _pos == pos+1)
        c[0] += 1
        c[other] -= 1
        succ.append(c)
    if y is not 0:
        c = list(config)
        other = next(i for i,_pos in enumerate(c) if _pos == pos-width)
        c[0] -= width
        c[other] += width
        succ.append(c)
    if y is not height-1:
        c = list(config)
        other = next(i for i,_pos in enumerate(c) if _pos == pos+width)
        c[0] += width
        c[other] -= width
        succ.append(c)
    return succ
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def plot_image(a,name):
        plt.figure(figsize=(6,6))
        plt.imshow(a,interpolation='nearest',cmap='gray',)
        plt.savefig(name)
    configs = generate_configs(6)
    plot_image(generate_puzzle(configs, 2, 3)[10],"puzzle.png")

    

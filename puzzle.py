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
        for y in range(height):
            for x in range(width):
                figure[y*base_height:(y+1)*base_height,
                       x*base_width:(x+1)*base_width] = panels[config[y*width+x]]
        return figure
    return np.array([ generate(c) for c in configs ]).reshape((-1,dim_y,dim_x))

# def puzzle_transitions(size=2,n = 10000):
#     configs = generate_configs(size,n)
#     def 

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def plot_image(a,name):
        plt.figure(figsize=(6,6))
        plt.imshow(a,interpolation='nearest',cmap='gray',)
        plt.savefig(name)
    configs = generate_configs(6)
    plot_image(generate_puzzle(configs, 2, 3)[10],"puzzle.png")

    

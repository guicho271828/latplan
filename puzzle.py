import numpy as np

panels = [
    [0, 0, 1, 0, 0,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 0, 1, 0, 0,],
    [0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,],
    [0, 0, 1, 0, 0,
     0, 1, 0, 1, 0,
     0, 0, 0, 1, 0,
     0, 0, 1, 0, 0,
     0, 1, 1, 1, 0,],
    [0, 1, 1, 1, 0,
     0, 0, 0, 1, 0,
     0, 1, 1, 1, 0,
     0, 0, 0, 1, 0,
     0, 1, 1, 1, 0,],
    [0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 1, 1, 0,
     0, 0, 0, 1, 0,
     0, 0, 0, 1, 0,],
    [0, 1, 1, 1, 0,
     0, 1, 0, 0, 0,
     0, 1, 1, 1, 0,
     0, 0, 0, 1, 0,
     0, 1, 1, 1, 0,],
    [0, 1, 1, 1, 0,
     0, 1, 0, 0, 0,
     0, 1, 1, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 1, 1, 0,],
    [0, 1, 1, 1, 0,
     0, 0, 0, 1, 0,
     0, 0, 0, 1, 0,
     0, 0, 0, 1, 0,
     0, 0, 0, 1, 0,],
    [0, 1, 1, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 1, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 1, 1, 0,],
    [0, 1, 1, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 1, 1, 0,
     0, 0, 0, 1, 0,
     0, 0, 0, 1, 0,],
    [0, 0, 0, 0, 0,
     0, 0, 0, 0, 0,
     0, 0, 1, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 1, 1, 0,],
    [0, 0, 0, 0, 0,
     0, 1, 0, 0, 0,
     0, 1, 1, 0, 0,
     0, 1, 0, 1, 0,
     0, 1, 1, 0, 0,],
    [0, 0, 0, 0, 0,
     0, 0, 1, 1, 0,
     0, 1, 0, 0, 0,
     0, 1, 0, 0, 0,
     0, 0, 1, 1, 0,],
    [0, 0, 0, 0, 0,
     0, 0, 0, 1, 0,
     0, 0, 1, 1, 0,
     0, 1, 0, 1, 0,
     0, 0, 1, 1, 0,],
    [0, 0, 1, 0, 0,
     0, 1, 0, 1, 0,
     0, 1, 1, 1, 0,
     0, 1, 0, 0, 0,
     0, 0, 1, 1, 0,],
    [0, 0, 0, 1, 0,
     0, 0, 1, 0, 0,
     0, 1, 1, 1, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,],
]

def generate_configs(digit=9):
    import itertools
    return itertools.permutations(range(digit))

def generate_mnist_puzzle(configs, width, height):
    assert width*height <= 9
    from mnist import mnist
    x_train, y_train, _, _ = mnist()
    mnist_filters = [ np.equal(i,y_train) for i in range(10) ]
    mnist_imgs    = [ x_train[f] for f in mnist_filters ]
    mnist_panels  = [ imgs[0].reshape((28,28)) for imgs in mnist_imgs ]
    dim_x = 28*width
    dim_y = 28*height
    def generate(config):
        figure = np.zeros((dim_y,dim_x))
        for y in range(height):
            for x in range(width):
                figure[y*28:(y+1)*28, x*28:(x+1)*28] = mnist_panels[config[y*width+x]]
        return figure
    return np.array([ generate(c) for c in configs ]).reshape((-1,dim_y,dim_x))

# def puzzle_transitions(size=2,n = 10000):
#     configs = generate_configs(size,n)
#     def 

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def plot_image(a,name):
        plt.figure(figsize=(6,6))
        plt.imshow(a)
        plt.savefig(name)
    configs = generate_configs(6)
    plot_image(generate_mnist_puzzle(configs, 2, 3)[10],"mnist-puzzle.png")

    

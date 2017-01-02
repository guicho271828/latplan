import numpy as np
from puzzle import generate_configs

def generate_mnist_puzzle(configs, width, height):
    assert width*height <= 9
    from mnist import mnist
    x_train, y_train, _, _ = mnist()
    filters = [ np.equal(i,y_train) for i in range(10) ]
    imgs    = [ x_train[f] for f in filters ]
    panels  = [ imgs[0].reshape((28,28)) for imgs in imgs ]
    base_width = 28
    base_height = 28
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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def plot_image(a,name):
        plt.figure(figsize=(6,6))
        plt.imshow(a)
        plt.savefig(name)
    configs = generate_configs(6)
    plot_image(generate_mnist_puzzle(configs, 2, 3)[10],"samples/mnist_puzzle.png")

    

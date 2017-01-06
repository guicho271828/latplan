#!/usr/bin/env python

import numpy as np
from model import GumbelAE

# if __name__ == '__main__':

def plot_grid(images,name="plan.png"):
    import matplotlib.pyplot as plt
    l = len(images)
    w = 32
    h = l//w+1
    plt.figure(figsize=(60, h*2))
    for i,image in enumerate(images):
        # display original
        ax = plt.subplot(h,w,i+1)
        plt.imshow(image,interpolation='nearest',cmap='gray',)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(name)

def plot_grid2(images,shape,w,name="plan.png"):
    import matplotlib.pyplot as plt
    l = images.shape[0]
    h = l//w
    margin = 3
    m_shape = (margin + np.array(shape))
    figure = np.ones(m_shape * np.array((h,w)))
    # print images.shape,h,w
    for y in range(h):
        for x in range(w):
            begin = m_shape * np.array((y,x))
            end   = (m_shape * (np.array((y,x))+1)) - margin
            # print begin,end,y*w+x
            figure[begin[0]:end[0],begin[1]:end[1]] = images[y*w+x]
    plt.figure(figsize=(h,w))
    plt.imshow(figure,interpolation='nearest',cmap='gray',)
    plt.savefig(name)

# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)

ae = GumbelAE("samples/puzzle3p_model/")
ae.build((12,15))
ae.load()
m = 16

zs = (((np.arange(2**m)[:,None] & (1 << np.arange(m)))) > 0).astype(int)
ys = ae.decode_binary(zs)

def run():
    per_image = 2**8
    for j in range((2**16) // per_image):
        # print "gathering..."
        # images = []
        # for z,y in zip(zs[j*per_image:(1+j)*per_image],
        #                ys[j*per_image:(1+j)*per_image]):
        #     images.append(z.reshape((4,4)))
        #     images.append(y)
        path = ae.local("all-bits{}.png".format(j))
        print path
        # plot_grid(images,path)
        # plot_grid(ys[j*per_image:(1+j)*per_image],path)
        plot_grid2(ys[j*per_image:(1+j)*per_image],(12,15),16,path)

run()

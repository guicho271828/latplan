import numpy as np

import math

def plot_grid(images,w=10,path="plan.png",verbose=False):
    import matplotlib.pyplot as plt
    l = 0
    l = len(images)
    h = int(math.ceil(l/w))
    plt.figure(figsize=(w*1.5, h*1.5))
    for i,image in enumerate(images):
        ax = plt.subplot(h,w,i+1)
        try:
            plt.imshow(image,interpolation='nearest',cmap='gray', vmin = 0, vmax = 1)
        except TypeError:
            TypeError("Invalid dimensions for image data: image={}".format(np.array(image).shape))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    print(path) if verbose else None
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def squarify(x,axis=-1):
    before, N, after = x.shape[:axis], x.shape[axis], x.shape[axis+1:]
    if axis == -1:
        after = tuple()
    import math
    l = math.ceil(math.sqrt(N))

    if l*l == N:
        return x.reshape((*before,l,l,*after))
    else:
        size = l*l
        return np.concatenate((x,np.ones((*before,size-N,*after))),axis=axis).reshape((*before,l,l,*after))


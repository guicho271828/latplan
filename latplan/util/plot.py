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

# contiguous image
def plot_grid2(images,w=10,path="plan.png",verbose=False):
    import matplotlib.pyplot as plt
    l = images.shape[0]
    h = int(math.ceil(l/w))
    margin = 3
    m_shape = (margin + np.array(images.shape[1:]))
    all_shape = m_shape * np.array((h,w))
    figure = np.ones(all_shape)
    print(images.shape,h,w,m_shape,figure.shape) if verbose else None
    for y in range(h):
        for x in range(w):
            begin = m_shape * np.array((y,x))
            end   = (m_shape * (np.array((y,x))+1)) - margin
            # print(begin,end,y*w+x)
            if y*w+x < len(images):
                figure[begin[0]:end[0],begin[1]:end[1]] = images[y*w+x]
    plt.figure(figsize=all_shape[::-1] * 0.01)
    plt.imshow(figure,interpolation='nearest',cmap='gray',)
    print(path) if verbose else None
    plt.tight_layout()
    plt.savefig(path)

def plot_ae(ae,data,path):
    return ae.plot(data,path)

def squarify(bitvectors):
    batch, N = bitvectors.shape
    import math
    root = math.sqrt(N)
    l1 = math.floor(root)

    if l1*l1 == N:
        return bitvectors.reshape((-1,l1,l1))
    else:
        l2 = math.ceil(root)
        size = l2*l2
        return np.concatenate((bitvectors,np.ones((bitvectors.shape[0],size-N))),axis=1).reshape((-1,l2,l2))
    
    

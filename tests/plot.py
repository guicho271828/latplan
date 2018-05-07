
import matplotlib.pyplot as plt

def plot_image(a,name):
    plt.figure(figsize=(20,20))
    plt.imshow(a,interpolation='nearest',cmap='gray',)
    plt.savefig(name)

import numpy as np

def fix_images(images,dims=None):
    if isinstance(images,list) or isinstance(images,tuple):
        expanded = []
        for i in images:
            expanded.extend(fix_image(i,dims))
        return expanded
    if len(images.shape) == 3:
        return images
    if len(images.shape) == 4:
        return np.einsum("bxyc->bcxy",images).reshape((-1,)+images.shape[1:3])
    if len(images.shape) == 2:
        return images.reshape((images.shape[0],)+dims)
    raise BaseException("images.shape={}, dims={}".format(images.shape,dims))

def fix_image(image,dims=None):
    if len(image.shape) == 2:
        return np.expand_dims(image,axis=0)
    if len(image.shape) == 3:
        return np.einsum("xyc->cxy",image).reshape((-1,)+image.shape[0:2])
    if len(image.shape) == 1:
        return image.reshape((1,)+dims)
    raise BaseException("image.shape={}, dims={}".format(image.shape,dims))

import math

def plot_grid(images,path="plan.png",w=6,verbose=False):
    import matplotlib.pyplot as plt
    l = 0
    images = fix_images(images)
    l = len(images)
    h = int(math.ceil(l/w))
    plt.figure(figsize=(w*8, h*8))
    for i,image in enumerate(images):
        ax = plt.subplot(h,w,i+1)
        try:
            plt.imshow(image,interpolation='nearest',cmap='gray',)
        except TypeError:
            TypeError("Invalid dimensions for image data: image={}".format(np.array(image).shape))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    print(path) if verbose else None
    plt.tight_layout()
    plt.savefig(path)


# def plot_grid(images,name="plan.png"):
#     l = len(images)
#     w = 6
#     h = max(l//6,1)
#     plt.figure(figsize=(20, h*2))
#     for i,image in enumerate(images):
#         # display original
#         ax = plt.subplot(h,w,i+1)
#         plt.imshow(image,interpolation='nearest',cmap='gray',)
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.savefig(name)

def puzzle_plot(p):
    p.setup()
    def name(template):
        return template.format(p.__name__)
    from itertools import islice
    configs = list(islice(p.generate_configs(9), 1000)) # be careful, islice is not immutable!!!
    import numpy.random as random
    random.shuffle(configs)
    configs = configs[:10]
    print(list(p.to_objects(configs,3,3)))
    puzzles = p.generate(configs, 3, 3)
    print(puzzles.shape, "mean", puzzles.mean(), "stdev", np.std(puzzles))
    plot_image(puzzles[-1], name("{}.png"))
    plot_image(np.clip(puzzles[-1]+np.random.normal(0,0.1,puzzles[-1].shape),0,1),name("{}+noise.png"))
    plot_image(np.round(np.clip(puzzles[-1]+np.random.normal(0,0.1,puzzles[-1].shape),0,1)),name("{}+noise+round.png"))
    plot_grid(puzzles, name("{}s.png"))
    _transitions = p.transitions(3,3,configs=configs)
    print(_transitions.shape)
    transitions_for_show = \
        np.einsum('ba...->ab...',_transitions) \
          .reshape((-1,)+_transitions.shape[2:])
    print(transitions_for_show.shape)
    plot_grid(transitions_for_show, name("{}_transitions.png"))

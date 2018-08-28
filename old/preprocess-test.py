#!/usr/bin/env python3

import numpy as np
import numpy.random as rn

with np.load("/u/jp576066/repos/clevr-blocksworld/blocks-3-3-32.npz") as data:
    im = data["images"]

im = im / 256
rn.shuffle(im)
im = im[:10].reshape((-1,32,32,3))
print(im.shape)
import imageio

def dump(im,name):
    print(name)
    import os.path
    if not os.path.isdir("preprocess"):
        os.makedirs("preprocess")
    for i,im in enumerate(im):
        imageio.imwrite("preprocess/{}-{:03d}.png".format(name,i),im*256)

def normalize(image):
    # into 0-1 range
    if image.max() == image.min():
        return image - image.min(), image.max(), image.min()
    else:
        return (image - image.min())/(image.max() - image.min()), image.max(), image.min()

def equalize(image):
    from skimage import exposure
    return exposure.equalize_hist(image)

def enhance(image):
    return np.clip((image-0.5)*3,-0.5,0.5)+0.5

def enhance2(image):
    tmp = image-0.5
    sgn = np.sign(tmp)
    return np.sqrt(np.abs(tmp))*sgn+0.5

def unnormalize(image,max,min):
    range = max-min
    return (image * range) + min

def unenhance(image):
    # not loss-less
    return (image - 0.5) / 3 + 0.5

def preprocess(image):
    image = equalize(image)
    image,_,_ = normalize(image)
    image = enhance(image)
    return image

dump(im,"original")
dump(enhance(im),"enhance")
dump(enhance2(im),"enhance2")
dump(equalize(im),"equalize")
dump(preprocess(im),"preprocess")

n,max,min = normalize(im)
dump(n,"normalize")
dump(unnormalize(n,max,min),"unnormalize")


dump([normalize(im)[0] for im in im],"normalize_perstate")

dump([ np.stack([normalize(im[:,:,0])[0],normalize(im[:,:,1])[0],normalize(im[:,:,2])[0]],axis=-1) for im in im],"normalize_percolor")


dump(enhance(n),"normalize-enhance")

dump(unenhance(enhance(im)),"unenhance")



import numpy as np
from scipy import misc

def split_image(path,width,height):
    img = misc.imread(path,True)/256
    # convert the image to *greyscale*
    W, H = img.shape
    dW, dH = W//width, H//height
    return np.array([
        img[dH*i:dH*(i+1), dH*j:dH*(j+1)]
        for i in range(width)
        for j in range(height) ])

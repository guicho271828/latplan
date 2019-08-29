
import numpy as np
import imageio

def split_image(path,width,height):
    img = imageio.imread(path,as_gray=True)/256
    # convert the image to *greyscale*
    W, H = img.shape
    dW, dH = W//width, H//height
    img = img[:height*dH,:width*dW]
    return np.transpose(img.reshape((height,dH,width,dW)), (0,2,1,3)).reshape((height*width,dH,dW))


import numpy as np

def gaussian(a, sigma=0.3):
    return np.random.normal(0,sigma,a.shape) + a



import numpy as np

def gaussian(a):
    return np.clip(np.random.normal(0,0.3,a.shape) + a, 0,1)

def salt(a):
    return np.clip(np.clip(np.sign(0.06 - np.random.uniform(0,1,a.shape)), 0, 1) + a, 0, 1)

def pepper(a):
    return np.clip(a - np.clip(np.sign(0.06 - np.random.uniform(0,1,a.shape)), 0, 1), 0, 1)


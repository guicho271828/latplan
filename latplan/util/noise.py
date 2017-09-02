
import numpy as np

def gaussian(a, sigma=0.3):
    return np.clip(np.random.normal(0,sigma,a.shape) + a, 0,1)

def salt(a,p=0.06):
    return np.clip(np.clip(np.sign(p - np.random.uniform(0,1,a.shape)), 0, 1) + a, 0, 1)

def pepper(a,p=0.06):
    return np.clip(a - np.clip(np.sign(p - np.random.uniform(0,1,a.shape)), 0, 1), 0, 1)

def saltpepper(a,p=0.06):
    def sym(x):
        return x*2-1
    def zeroone(x):
        return (x+1)/2
    return zeroone(sym(a) * sym(np.clip(np.sign(np.random.uniform(0,1,a.shape) - p), 0, 1)))



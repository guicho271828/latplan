from . import trace
from . import plot
from . import mnist

def get_ae_type(directory):
    import os.path
    d, n = os.path.split(directory)
    if n == '':
        return d.split("/")[-1].split("_")[-1]
    else:
        return n.split("_")[-1]
    

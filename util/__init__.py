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
    

def curry(fn,*args1,**kwargs1):
    return lambda *args,**kwargs: fn(*args1,*args,**{**kwargs1,**kwargs})

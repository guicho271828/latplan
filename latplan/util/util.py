def curry(fn,*args1,**kwargs1):
    return lambda *args,**kwargs: fn(*args1,*args,**{**kwargs1,**kwargs})

def prepare_binary_classification_data(real, fake):
    import numpy as np
    shape = real.shape[1:]
    
    both = np.concatenate((real, fake),axis=0)
    both = np.reshape(both, (len(both), -1)) # flatten

    data_dim = both.shape[1]

    both2 = np.pad(both, ((0,0),(0,1)), 'constant') # default 0
    both2[:len(real),-1] = 1                        # tag true
    np.random.shuffle(both2)

    train_in  = np.reshape(both2[:int(0.9*len(both2)), :data_dim], (-1, *shape))
    train_out = both2[:int(0.9*len(both2)), -1]
    test_in   = np.reshape(both2[int(0.9*len(both2)):, :data_dim], (-1, *shape))
    test_out  = both2[int(0.9*len(both2)):, -1]

    return train_in, train_out, test_in, test_out


import numpy as np
def set_difference(a, b):
    assert a.shape[1:] == b.shape[1:]
    a = a.copy()
    b = b.copy()
    a_v = a.view([('', a.dtype)] * a.shape[1])
    b_v = b.view([('', b.dtype)] * b.shape[1])
    return np.setdiff1d(a_v, b_v).view(a.dtype).reshape((-1, a.shape[1]))

def union(a, b):
    assert a.shape[1:] == b.shape[1:]
    d = set_difference(a, b)
    return np.concatenate((b, d), axis=0)


def ensure_list(x):
    if type(x) is not list:
        return [x]
    else:
        return x

import json
import numpy as np
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def parse_desc(desc):
    result = {}
    for each in desc.split(","):
        key, value = each.strip().split(": ")
        result[key] = value
    return result


def gpu_info():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    result = []
    for x in local_device_protos:
        if x.device_type == 'GPU':
            result.append(parse_desc(x.physical_device_desc))
    return result

#!/usr/bin/env python3
import warnings
import config
import sys
import numpy as np
import latplan
import latplan.model
from latplan.model       import combined_sd
from latplan.util        import curry, set_difference, prepare_binary_classification_data
from latplan.util.tuning import grid_search, nn_task
from latplan.util.np_distances import *
import numpy.random as random
import keras.backend as K
import tensorflow as tf

from keras.optimizers import Adam
from keras_adabound   import AdaBound
from keras_radam      import RAdam

import keras.optimizers

setattr(keras.optimizers,"radam", RAdam)
setattr(keras.optimizers,"adabound", AdaBound)

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(threshold=sys.maxsize,formatter={'float_kind':float_formatter})

################################################################

sae = None
aae = None
cae = None
sd3 = None
discriminator = None

################################################################
# training parameters

default_parameters = {
    'lr'              : 0.0001,
    'batch_size'      : 2000,
    'full_epoch'      : 1000,
    'epoch'           : 1000,
    'max_temperature' : 2.0,
    'min_temperature' : 0.1,
    'M'               : 2,
    'min_grad'        : 0.0,
    'optimizer'       : 'radam',
}

# exhaustive tuning
parameters = {
    'num_layers' :[1,2,3],
    'layer'      :[50,300,1000],# [400,4000],
    'dropout'    :[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],    #[0.1,0.4], #0.6,0.7,
    'batch_size' :[1000],
    'full_epoch' :[1000],
    'activation' :['tanh','relu'],
    # quick eval
    'epoch'      :[3000],
    'lr'         :[0.001],
}

# tuned results
parameters = {
    'num_layers' :[1,2],
    'layer'      :[300],# [400,4000],
    'dropout'    :[0.5, 0.8],    #[0.1,0.4], #0.6,0.7,
    'batch_size' :[1000],
    'full_epoch' :[400],
    'activation' :['relu'],
    # quick eval
    'epoch'      :[400],
    'lr'         :[0.01,0.001],
}

def learn(path):
    network = latplan.model.get('PUDiscriminator')
    true_actions = np.loadtxt(sae.local("actions.csv"),dtype=np.int8)
    fake_actions = np.loadtxt(aae.local("fake_actions.csv"),dtype=np.int8)
    train_in, train_out, val_in, val_out = prepare_binary_classification_data(true_actions, fake_actions)
    discriminator,_,_ = grid_search(curry(nn_task, network, path,
                                          train_in, train_out, val_in, val_out,),
                                    default_parameters,
                                    parameters,
                                    path,
    )
    discriminator.save()
    return discriminator

def test():
    valid = np.loadtxt(aae.local("valid_actions.csv"),dtype=np.int8)
    random.shuffle(valid)

    invalid = np.loadtxt(aae.local("invalid_actions.csv"),dtype=np.int8)
    random.shuffle(invalid)

    N = int(valid.shape[1] // 2)
    
    performance = {}
    def reg(names,value,d=performance):
        name = names[0]
        if len(names)>1:
            try:
                tmp = d[name]
            except KeyError:
                tmp={}
                d[name]=tmp
            reg(names[1:], value, tmp)
        else:
            d[name] = float(value)
            print(name,": ", value)
    
    reg(["valid"],   len(valid))
    reg(["invalid"], len(invalid))

    def measure(valid, invalid, suffix):
        minlen=min(len(valid),len(invalid))
        
        valid_tmp   = valid  [:minlen]
        invalid_tmp = invalid[:minlen]

        tp = np.clip(discriminator.discriminate(  valid_tmp,batch_size=1000).round(), 0,1) # true positive
        fp = np.clip(discriminator.discriminate(invalid_tmp,batch_size=1000).round(), 0,1) # false positive
        tn = 1-fp
        fn = 1-tp
    
        reg([suffix,"minlen"     ],minlen)
        recall      = np.mean(tp) # recall / sensitivity / power / true positive rate out of condition positive
        specificity = np.mean(tn) # specificity / true negative rate out of condition negative
        reg([suffix,"recall"     ],recall)
        reg([suffix,"specificity"],specificity)
        reg([suffix,"f"],(2*recall*specificity)/(recall+specificity))
        try:
            reg([suffix,"precision"  ],np.sum(tp)/(np.sum(tp)+np.sum(fp)))
        except ZeroDivisionError:
            reg([suffix,"precision"  ],float('nan'))
        try:
            reg([suffix,"accuracy"   ],(np.sum(tp)+np.sum(tn))/(2*minlen))
        except ZeroDivisionError:
            reg([suffix,"accuracy"   ],float('nan'))
        return

    measure(valid,invalid,"raw")
    measure(valid,invalid[np.where(np.squeeze(combined_sd(invalid[:,N:],sae,cae,sd3,batch_size=1000)) > 0.5)[0]],     "sd")
    
    p = latplan.util.puzzle_module(sae.local(""))
    measure(valid,invalid[p.validate_states(sae.decode(invalid[:,N:],batch_size=1000),verbose=False,batch_size=1000)],"validated")
    
    import json
    with open(discriminator.local('performance.json'), 'w') as f:
        json.dump(performance, f)

def main(directory, mode="test", _aae="_aae"):
    global sae, aae, sd3, discriminator
    sae = latplan.model.load(directory)
    aae = latplan.model.load(sae.local(_aae))
    cae = latplan.model.load(sae.local("_cae/"),allow_failure=True)
    sd3 = latplan.model.load(sae.local("_sd3/"))

    if 'learn' in mode:
        discriminator = learn(sae.local(_aae+"_ad/"))
    else:
        discriminator = latplan.model.load(sae.local(_aae+"_ad/"))

    if 'test' in mode:
        test()

if __name__ == '__main__':
    import sys
    print(sys.argv)
    try:
        main(*sys.argv[1:])
    except:
        import latplan.util.stacktrace
        latplan.util.stacktrace.format()



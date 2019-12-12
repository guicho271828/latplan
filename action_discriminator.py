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
# random action generators
    
inflation = 1

def generate_nop(data):
    dim = data.shape[1]//2
    pre, suc = data[:,:dim], data[:,dim:]
    pre = np.concatenate((pre, suc), axis=0)
    data_invalid = np.concatenate((pre,pre),axis=1)
    data_invalid = set_difference(data_invalid, data)
    return data_invalid

def permute_suc(data):
    dim = data.shape[1]//2
    pre, suc = data[:,:dim], data[:,dim:]
    suc_invalid = np.copy(suc)
    random.shuffle(suc_invalid)
    data_invalid = np.concatenate((pre,suc_invalid),axis=1)
    data_invalid = set_difference(data_invalid, data)
    return data_invalid

def generate_random_action(data, sae):
    # reconstructable, maybe invalid
    dim = data.shape[1]//2
    pre, suc = data[:,:dim], data[:,dim:]
    from state_discriminator3 import generate_random
    pre = np.concatenate((pre, suc), axis=0)
    suc = np.concatenate((generate_random(pre, sae),
                          generate_random(pre, sae)), axis=0)[:len(pre)]
    actions_invalid = np.concatenate((pre, suc), axis=1)
    actions_invalid = set_difference(actions_invalid, data)
    return actions_invalid

def generate_random_action2(data):
    # completely random strings
    return np.random.randint(0,2,data.shape,dtype=np.int8)

def generate_aae_action():
    return np.loadtxt(sae.local("fake_actions.csv"),dtype=np.int8)

################################################################
# data preparation

# discriminate correct transitions and nop, suc-permutation, reconstructable, and random bits combined
# **** does not discriminate OEA-generated states quite well, do not use ****
def prepare(data):
    print("discriminate correct transitions and nop, suc-permutation, reconstructable, and random bits combined")
    print("**** does not discriminate OEA-generated states quite well, do not use ****")
    data_invalid = np.concatenate(
        tuple([generate_nop(data),
               *[ permute_suc(data) for i in range(inflation) ],
               *[ generate_random_action(data, sae) for i in range(inflation) ],
               *[ generate_random_action2(data) for i in range(inflation) ]
        ]), axis=0)

    data_valid   = np.repeat(data, len(data_invalid)//len(data), axis=0)

    return (latplan.model.get('PUDiscriminator'), *prepare_binary_classification_data(data_valid, data_invalid))

# This is a cheating, since we assume validation oracle
def prepare_aae_validated(known_transitions):
    print("generate many actions from states using OEA (at least one action for each state is correct)",
          "validate it with validators, then discriminate the correct vs wrong transitions.",
          sep="\n")
    print("**** CHEATING ****")
    N = known_transitions.shape[1] // 2
    y = generate_aae_action()

    p = latplan.util.puzzle_module(sae.path)
    batch = 100000
    valid_suc = np.zeros(len(y),dtype=bool)
    for i in range(1+len(y)//batch):
        print(i,"/",len(y)//batch)
        suc_images = sae.decode(y[batch*i:batch*(i+1),N:],batch_size=1000)
        valid_suc[batch*i:batch*(i+1)] = p.validate_states(suc_images,verbose=False,batch_size=1000)
        # This state validation is just for reducing the later effort for validating transitions
    
    before_len = len(y)
    y = y[valid_suc]
    print("removing invalid successor states:",before_len,"->",len(y))

    answers = np.zeros(len(y),dtype=int)
    for i in range(1+len(y)//batch):
        print(i,"/",len(y)//batch)
        pre_images = sae.decode(y[batch*i:batch*(i+1),:N],batch_size=1000)
        suc_images = sae.decode(y[batch*i:batch*(i+1),N:],batch_size=1000)
        answers[batch*i:batch*(i+1)] = np.array(p.validate_transitions([pre_images, suc_images],batch_size=1000)).astype(int)
    
    l = len(y)
    positive = np.count_nonzero(answers)
    print(positive,l-positive)

    y_positive = y[answers.astype(bool)]
    y_negative = y[(1-answers).astype(bool)]
    y_negative = y_negative[:len(y_positive)]
    
    return (latplan.model.get('Discriminator'), *prepare_binary_classification_data(y_positive, y_negative))

# discriminate correct transitions and other transitions generated by AAE
def prepare_aae_PU(known_transitions):
    print("discriminate correct transitions and other transitions generated by AAE")
    y = generate_aae_action()
    # normalize
    y = y[:len(known_transitions)]
    return (latplan.model.get('PUDiscriminator'), *prepare_binary_classification_data(known_transitions, y))

# discriminate the correct transitions and the other transitions generated by AAE,
# filtered by the state validator ***CHEATING***
def prepare_aae_PU2(known_transitions):
    print("discriminate the correct transitions and the other transitions generated by AAE, filtered by the state validator")
    print("**** CHEATING ****")
    N = known_transitions.shape[1] // 2
    y = generate_aae_action()
    p = latplan.util.puzzle_module(sae.path)
    batch = 100000
    valid_suc = np.zeros(len(y),dtype=bool)
    for i in range(1+len(y)//batch):
        print(i,"/",len(y)//batch)
        suc_images = sae.decode(y[batch*i:batch*(i+1),N:],batch_size=1000)
        valid_suc[batch*i:batch*(i+1)] = p.validate_states(suc_images,verbose=False,batch_size=1000)
    
    before_len = len(y)
    y = y[valid_suc]
    print("removing invalid successor states:",before_len,"->",len(y))
    y = y[:len(known_transitions)]
    # normalize
    return (latplan.model.get('PUDiscriminator'), *prepare_binary_classification_data(known_transitions, y))

# discriminate the correct transitions and the other transitions generated by AAE,
# filtered by the learned state discriminator
def prepare_aae_PU3(known_transitions):
    print("discriminate the correct transitions and the other transitions generated by AAE,",
          " filtered by the learned state discriminator",
          sep="\n")
    N = known_transitions.shape[1] // 2
    y = generate_aae_action()

    print("removing invalid successors (sd3)")
    ind = np.where(np.squeeze(combined_sd(y[:,N:],sae,cae,sd3,batch_size=1000)) > 0.5)[0]
    
    y = y[ind]
    if len(known_transitions) > 100:
        y = y[:len(known_transitions)] # undersample
    
    print("valid:",len(known_transitions),"mixed:",len(y),)
    print("creating binary classification labels")
    return (latplan.model.get('PUDiscriminator'), *prepare_binary_classification_data(known_transitions, y))


# discriminate the correct transitions and the other transitions generated by AAE,
# filtered by the learned state discriminator.
# The input is state-action pairs.
def prepare_aae_PU4(known_transitions):
    print("discriminate the correct transitions and the other transitions generated by AAE,",
          " filtered by the learned state discriminator",
          "The input is state-action pairs.",
          sep="\n")
    N = known_transitions.shape[1] // 2
    fake_transitions = generate_aae_action()

    print("removing invalid successors (sd3)")
    ind = np.where(np.squeeze(combined_sd(fake_transitions[:,N:],sae,cae,sd3,batch_size=1000)) > 0.5)[0]
    
    fake_transitions = fake_transitions[ind]
    if len(known_transitions) > 100:
        fake_transitions = fake_transitions[:len(known_transitions)] # undersample
    
    print("valid:",len(known_transitions),"mixed:",len(fake_transitions),)
    print("limit the input to the current state")
    known_actions  = aae.encode_action(known_transitions, batch_size=1000).round().reshape((-1, aae.parameters["M"]))
    fake_actions   = aae.encode_action(fake_transitions,  batch_size=1000).round().reshape((-1, aae.parameters["M"]))
    
    known_pre = known_transitions[:, :N]
    fake_pre  = fake_transitions[:, :N]
    
    known_input = np.concatenate((known_pre,known_actions), axis=1)
    fake_input = np.concatenate((fake_pre,fake_actions), axis=1)
    
    print("creating binary classification labels")
    return (latplan.model.get('PUDiscriminator'), *prepare_binary_classification_data(known_input, fake_input))

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

# good for puzzles
# {"dropout": 0.8, "full_epoch": 1000, "layer": 300, "num_layers": 1,
#  "batch_size": 1000, "activation": "relu", "epoch": 3000, "lr": 0.001}
# good for lightsout
# {'dropout': 0.5, 'full_epoch': 1000, 'layer': 300, 'num_layers': 2,
#  'batch_size': 1000, 'activation': 'relu', 'epoch': 3000, 'lr': 0.001}

def learn(input_type,subdir):
    if "hanoi" in sae.path:
        data = np.loadtxt(sae.local("all_actions.csv"),dtype=np.int8)
    else:
        data = np.loadtxt(sae.local("actions.csv"),dtype=np.int8)
    network, train_in, train_out, test_in, test_out = eval(input_type)(data)
    default_parameters["input_type"] = input_type
    discriminator,_,_ = grid_search(curry(nn_task, network, sae.local(subdir),
                                          train_in, train_out, test_in, test_out,),
                                    default_parameters,
                                    parameters)
    discriminator.save()
    return discriminator

def test():
    valid = np.loadtxt(sae.local("valid_actions.csv"),dtype=np.int8)
    random.shuffle(valid)

    invalid = np.loadtxt(sae.local("invalid_actions.csv"),dtype=np.int8)
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

        if discriminator.parameters["input_type"] == "prepare_aae_PU4":
            valid_actions   = aae.encode_action(valid_tmp, batch_size=1000).round().reshape((-1, aae.parameters["M"]))
            invalid_actions = aae.encode_action(invalid_tmp,  batch_size=1000).round().reshape((-1, aae.parameters["M"]))
            valid_pre       = valid_tmp[:, :N]
            invalid_pre     = invalid_tmp[:, :N]
            valid_tmp       = np.concatenate((valid_pre,valid_actions), axis=1)
            invalid_tmp     = np.concatenate((invalid_pre,invalid_actions), axis=1)

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

def main(directory, mode="test", input_type="prepare_aae_PU3", subdir="_ad/"):
    global sae, aae, sd3, discriminator
    sae = latplan.model.load(directory)
    aae = latplan.model.load(sae.local("_aae/"))
    cae = latplan.model.load(sae.local("_cae/"),allow_failure=True)
    sd3 = latplan.model.load(sae.local("_sd3/"))

    if 'learn' in mode:
        discriminator = learn(input_type,subdir)
    else:
        discriminator = latplan.model.load(sae.local(subdir))

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



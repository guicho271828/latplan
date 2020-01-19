#!/usr/bin/env python3
import warnings
import config
import sys
import numpy as np
import latplan.model
from latplan.model import ActionAE, CubeActionAE
from latplan.util        import curry, set_difference
from latplan.util.tuning import grid_search, nn_task, simple_genetic_search, reproduce

import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(threshold=sys.maxsize,formatter={'float_kind':float_formatter})

from keras.optimizers import Adam
from keras_adabound   import AdaBound
from keras_radam      import RAdam

import keras.optimizers

setattr(keras.optimizers,"radam", RAdam)
setattr(keras.optimizers,"adabound", AdaBound)

################################################################

# default values
default_parameters = {
    'epoch'           : 200,
    'batch_size'      : 500,
    'optimizer'       : "radam",
    'max_temperature' : 5.0,
    'min_temperature' : 0.7,
    'test_gumbel'     : False,
}

parameters = {
    'M'          :[50,100,200,400,800,1600],
    'N'          :[1],
    'dropout'    :[0.4],
    'aae_width'      :[100,300,600,],
    'aae_depth'      :[0,1,2],
    'aae_activation' :['relu','tanh'],
    'lr'         :[0.1,0.01,0.001],
}

import numpy.random as random

import sys
if len(sys.argv) == 1:
    sys.exit("{} [directory]".format(sys.argv[0]))

directory = sys.argv[1]
mode      = sys.argv[2]
aeclass   = sys.argv[3]
num_actions = eval(sys.argv[4])

sae = latplan.model.load(directory)

data = np.loadtxt(sae.local("actions.csv"),dtype=np.int8)

print(data.shape)
N = data.shape[1]//2
train = data[:int(len(data)*0.9)]
val   = data[int(len(data)*0.9):int(len(data)*0.95)]
test  = data[int(len(data)*0.95):]

if 'learn' in mode:
    print("start training")
    if num_actions is not None:
        parameters['M'] = [num_actions]
    aae,_,_ = simple_genetic_search(
        curry(nn_task, eval(aeclass), sae.local("_{}_{}/".format(aeclass,num_actions)), train, train, val, val,),
        default_parameters,
        parameters,
        sae.local("_{}_{}/".format(aeclass,num_actions)),
        limit=100,
        report_best= lambda net: net.save(),
    )
elif 'reproduce' in mode:
    aae,_,_ = reproduce(
        curry(nn_task, eval(aeclass), sae.local("_{}_{}/".format(aeclass,num_actions)), train, train, val, val,),
        default_parameters,
        parameters,
        sae.local("_{}_{}/".format(aeclass,num_actions)),)
    aae.save()
else:
    aae = eval(aeclass)(sae.local("_{}_{}/".format(aeclass,num_actions))).load()

num_actions = aae.parameters["M"]

actions = aae.encode_action(data, batch_size=1000).round()
histogram = np.squeeze(actions.sum(axis=0,dtype=int))
print(histogram)
print(np.count_nonzero(histogram > 0))
all_labels = np.zeros((np.count_nonzero(histogram), actions.shape[1], actions.shape[2]), dtype=int)
for i, pos in enumerate(np.where(histogram > 0)[0]):
    all_labels[i][0][pos] = 1

if 'plot' in mode:
    aae.plot(train[:8], "aae_train.png")
    aae.plot(test[:8], "aae_test.png")
    
    
    aae.plot(train[:8], "aae_train_decoded.png", sae=sae)
    aae.plot(test[:8], "aae_test_decoded.png", sae=sae)
    
    transitions = aae.decode([np.repeat(data[:1,:N], len(all_labels), axis=0), all_labels])
    aae.plot(transitions, "aae_all_actions_for_a_state.png", sae=sae)
    
    from latplan.util.timer import Timer
    # with Timer("loading csv..."):
    #     all_actions = np.loadtxt("{}/all_actions.csv".format(directory),dtype=np.int8)
    # transitions = aae.decode([np.repeat(all_actions[:1,:N], len(all_labels), axis=0), all_labels])
    suc = transitions[:,N:]
    from latplan.util.plot import plot_grid, squarify
    plot_grid([x for x in sae.decode(suc)], w=8, path=aae.local("aae_all_actions_for_a_state_8x16.png"), verbose=True)
    plot_grid([x for x in sae.decode(suc)], w=16, path=aae.local("aae_all_actions_for_a_state_16x8.png"), verbose=True)
    plot_grid(sae.decode(data[:1,:N]), w=1, path=aae.local("aae_all_actions_for_a_state_state.png"), verbose=True)
    

if 'test' in mode:
    from latplan.util.timer import Timer

    # note: unlike rf, product of bitwise match probability is not grouped by actions
    performance = {}
    performance["mae"]           = {} # average bitwise match
    performance["prob_bitwise"]  = {} # product of bitwise match probabilty
    performance["prob_allmatch"] = {} # probability of complete match
    
    def metrics(data,track):
        data_match           = 1-np.abs(aae.autoencode(data)-data)[:,N:]
        performance["mae"][track]           = float(np.mean(1-data_match))                 # average bitwise match
        performance["prob_bitwise"][track]  = float(np.prod(np.mean(data_match,axis=0))) # product of bitwise match probabilty
        performance["prob_allmatch"][track] = float(np.mean(np.prod(data_match,axis=1))) # probability of complete match

    metrics(val,"val")
    metrics(train,"train")
    metrics(test,"test")

    performance["effective_labels"] = int(len(all_labels))

    import json
    with open(aae.local("performance.json"),"w") as f:
        json.dump(performance, f)

def generate_aae_action(known_transisitons):
    N = known_transisitons.shape[1] // 2
    states = known_transisitons.reshape(-1, N)
    
    def repeat_over(array, repeats, axis=0):
        array = np.expand_dims(array, axis)
        array = np.repeat(array, repeats, axis)
        return np.reshape(array,(*array.shape[:axis],-1,*array.shape[axis+2:]))
    
    print("start generating transitions")
    random_actions = all_labels[np.random.choice(len(all_labels), len(states))]
    
    y = aae.decode([states, random_actions], batch_size=1000).round().astype(np.int8)

    print("remove known transitions")
    y = set_difference(y, known_transisitons)
    print("shuffling")
    random.shuffle(y)
    return y

if "dump" in mode:
    # dump list of available actions
    print(aae.local("available_actions.csv"))
    with open(aae.local("available_actions.csv"), 'wb') as f:
        np.savetxt(f,np.where(histogram > 0)[0],"%d")

    # one-hot to id
    actions_byid = (actions * np.arange(num_actions)).sum(axis=-1,dtype=int)
    print(aae.local("actions+ids.csv"))
    with open(aae.local("actions+ids.csv"), 'wb') as f:
        np.savetxt(f,np.concatenate((data,actions_byid), axis=1),"%d")

    # note: fake_transitions are already shuffled, and also do not contain any examples in data.
    fake_transitions  = generate_aae_action(data)
    fake_actions      = aae.encode_action(fake_transitions, batch_size=1000).round()
    fake_actions_byid = (fake_actions * np.arange(num_actions)).sum(axis=-1,dtype=int)

    print(aae.local("fake_actions.csv"))
    with open(aae.local("fake_actions.csv"), 'wb') as f:
        np.savetxt(f,fake_transitions,"%d")
    print(aae.local("fake_actions+ids.csv"))
    with open(aae.local("fake_actions+ids.csv"), 'wb') as f:
        np.savetxt(f,np.concatenate((fake_transitions,fake_actions_byid), axis=1),"%d")
    
    test_transitions  = generate_aae_action(data)
    test_actions      = aae.encode_action(test_transitions, batch_size=1000).round()
    test_actions_byid = (test_actions * np.arange(num_actions)).sum(axis=-1,dtype=int)

    p = latplan.util.puzzle_module(sae.path)
    print("decoding pre")
    pre_images = sae.decode(test_transitions[:,:N],batch_size=1000)
    print("decoding suc")
    suc_images = sae.decode(test_transitions[:,N:],batch_size=1000)
    print("validating transitions")
    valid   = p.validate_transitions([pre_images, suc_images],batch_size=1000)
    invalid = np.logical_not(valid)
    
    valid_transitions    = test_transitions [valid]
    valid_actions_byid   = test_actions_byid[valid]
    invalid_transitions  = test_transitions [invalid]
    invalid_actions_byid = test_actions_byid[invalid]

    print(aae.local("valid_actions.csv"))
    with open(aae.local("valid_actions.csv"), 'wb') as f:
        np.savetxt(f,valid_transitions,"%d")
    print(aae.local("valid_actions+ids.csv"))
    with open(aae.local("valid_actions+ids.csv"), 'wb') as f:
        np.savetxt(f,np.concatenate((valid_transitions,valid_actions_byid), axis=1),"%d")
        
    print(aae.local("invalid_actions.csv"))
    with open(aae.local("invalid_actions.csv"), 'wb') as f:
        np.savetxt(f,invalid_transitions,"%d")
    print(aae.local("invalid_actions+ids.csv"))
    with open(aae.local("invalid_actions+ids.csv"), 'wb') as f:
        np.savetxt(f,np.concatenate((invalid_transitions,invalid_actions_byid), axis=1),"%d")


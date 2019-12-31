#!/usr/bin/env python3
import warnings
import config
import sys
import numpy as np
import latplan.model
from latplan.model import ActionAE
from latplan.util        import curry, set_difference
from latplan.util.tuning import grid_search, nn_task

import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(threshold=sys.maxsize,formatter={'float_kind':float_formatter})

################################################################

# default values
default_parameters = {
    'lr'              : 0.0001,
    'batch_size'      : 2000,
    'full_epoch'      : 1000,
    'epoch'           : 1000,
    'max_temperature' : 5.0,
    'min_temperature' : 0.1,
    'M'               : 2,
    'argmax'          : True,
}

min_num_actions = 8
max_num_actions = 128
inc_num_actions = 8

import numpy.random as random

import sys
if len(sys.argv) == 1:
    sys.exit("{} [directory]".format(sys.argv[0]))

directory = sys.argv[1]
mode      = sys.argv[2]

sae = latplan.model.load(directory)

if "hanoi" in sae.path:
    data = np.loadtxt(sae.local("all_actions.csv"),dtype=np.int8)
else:
    data = np.loadtxt(sae.local("actions.csv"),dtype=np.int8)

print(data.shape)
N = data.shape[1]//2
train = data[:int(len(data)*0.9)]
val   = data[int(len(data)*0.9):int(len(data)*0.95)]
test  = data[int(len(data)*0.95):]

try:
    if 'learn' in mode:
        raise Exception('learn')
    aae = ActionAE(sae.local("_aae/")).load()
    num_actions = aae.parameters["M"]
except:
    print("start training")
    for num_actions in range(min_num_actions,max_num_actions,inc_num_actions):
        parameters = {
            'N'          :[1],
            'M'          :[num_actions],
            'layer'      :[400],# 200,300,400,700,1000
            'encoder_layers' : [2], # 0,2,3
            'decoder_layers' : [2], # 0,1,3
            'dropout'    :[0.4], #[0.1,0.4],
            # 'dropout_z'  :[False],
            'batch_size' :[2000],
            'full_epoch' :[1000],
            'epoch'      :[1000],
            'encoder_activation' :['relu'], # 'tanh'
            'decoder_activation' :['relu'], # 'tanh',
            # quick eval
            'lr'         :[0.001],
            }
        aae,_,_ = grid_search(curry(nn_task, ActionAE, sae.local("_aae/"), train, train, val, val,),
                              default_parameters,
                              parameters)
        val_diff = np.abs(aae.autoencode(val)-val)[:,N:]
        val_mae  = np.mean(val_diff)
        
        if val_mae < (1.0 / N): # i.e. below 1 bit
            break
    aae.save()

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
    # s1,s2,s3,s1,s2,s3,....
    repeated_states  = repeat_over(states, len(all_labels), axis=0)
    # a1,a1,a1,a2,a2,a2,....
    repeated_actions = np.repeat(all_labels, len(states), axis=0)
    
    y = aae.decode([repeated_states, repeated_actions], batch_size=1000).round().astype(np.int8)

    print("remove known transitions")
    y = set_difference(y, known_transisitons)
    print("shuffling")
    random.shuffle(y)
    return y

if "dump" in mode:
    # dump list of available actions
    print(sae.local("available_actions.csv"))
    with open(sae.local("available_actions.csv"), 'wb') as f:
        np.savetxt(f,np.where(histogram > 0)[0],"%d")

    # one-hot to id
    actions_byid = (actions * np.arange(num_actions)).sum(axis=-1,dtype=int)
    print(sae.local("actions+ids.csv"))
    with open(sae.local("actions+ids.csv"), 'wb') as f:
        np.savetxt(f,np.concatenate((data,actions_byid), axis=1),"%d")

    transitions = generate_aae_action(data)
    # note: transitions are already shuffled, and also do not contain any examples in data.
    actions      = aae.encode_action(transitions, batch_size=1000).round()
    actions_byid = (actions * np.arange(num_actions)).sum(axis=-1,dtype=int)

    # ensure there are enough test examples
    separation = min(len(data)*10,len(transitions)-len(data))
    
    # fake dataset is used only for the training.
    fake_transitions  = transitions[:separation]
    fake_actions_byid = actions_byid[:separation]

    # remaining data are used only for the testing.
    test_transitions  = transitions[separation:]
    test_actions_byid = actions_byid[separation:]

    print(fake_transitions.shape, test_transitions.shape)

    print(sae.local("fake_actions.csv"))
    with open(sae.local("fake_actions.csv"), 'wb') as f:
        np.savetxt(f,fake_transitions,"%d")
    print(sae.local("fake_actions+ids.csv"))
    with open(sae.local("fake_actions+ids.csv"), 'wb') as f:
        np.savetxt(f,np.concatenate((fake_transitions,fake_actions_byid), axis=1),"%d")
    
    p = latplan.util.puzzle_module(sae.path)
    print("decoding pre")
    pre_images = sae.decode(test_transitions[:,:N],batch_size=1000)
    print("decoding suc")
    suc_images = sae.decode(test_transitions[:,N:],batch_size=1000)
    print("validating transitions")
    valid    = p.validate_transitions([pre_images, suc_images],batch_size=1000)
    invalid  = np.logical_not(valid)
    
    valid_transitions  = test_transitions [valid][:len(data)] # reduce the amount of data to reduce runtime
    valid_actions_byid = test_actions_byid[valid][:len(data)]
    invalid_transitions  = test_transitions [invalid][:len(data)] # reduce the amount of data to reduce runtime
    invalid_actions_byid = test_actions_byid[invalid][:len(data)]

    print(sae.local("valid_actions.csv"))
    with open(sae.local("valid_actions.csv"), 'wb') as f:
        np.savetxt(f,valid_transitions,"%d")
    print(sae.local("valid_actions+ids.csv"))
    with open(sae.local("valid_actions+ids.csv"), 'wb') as f:
        np.savetxt(f,np.concatenate((valid_transitions,valid_actions_byid), axis=1),"%d")
        
    print(sae.local("invalid_actions.csv"))
    with open(sae.local("invalid_actions.csv"), 'wb') as f:
        np.savetxt(f,invalid_transitions,"%d")
    print(sae.local("invalid_actions+ids.csv"))
    with open(sae.local("invalid_actions+ids.csv"), 'wb') as f:
        np.savetxt(f,np.concatenate((invalid_transitions,invalid_actions_byid), axis=1),"%d")


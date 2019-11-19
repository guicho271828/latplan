#!/usr/bin/env python3
import warnings
import config
import sys
import numpy as np
import latplan.model
from latplan.model import ActionAE
from latplan.util        import curry
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
val   = data[int(len(data)*0.9):]

try:
    if 'learn' in mode:
        raise Exception('learn')
    aae = ActionAE(sae.local("_aae/")).load()
    num_actions = aae.parameters["M"]
except:
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
effective_labels = np.count_nonzero(histogram)
all_labels = np.zeros((effective_labels, actions.shape[1], actions.shape[2]), dtype=int)
for i, pos in enumerate(np.where(histogram > 0)[0]):
    all_labels[i][0][pos] = 1

if 'plot' in mode:
    aae.plot(data[:8], "aae_train.png")
    aae.plot(data[int(len(data)*0.9):int(len(data)*0.9)+8], "aae_test.png")
    
    
    aae.plot(data[:8], "aae_train_decoded.png", sae=sae)
    aae.plot(data[int(len(data)*0.9):int(len(data)*0.9)+8], "aae_test_decoded.png", sae=sae)
    
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
    with Timer("loading csv..."):
        all_actions = np.loadtxt("{}/all_actions.csv".format(directory),dtype=np.int8)

    # note: unlike rf, product of bitwise match probability is not grouped by actions
    performance = {}
    performance["mae"]           = {} # average bitwise match
    performance["prob_bitwise"]  = {} # product of bitwise match probabilty
    performance["prob_allmatch"] = {} # probability of complete match
    
    def metrics(data,track):
        data_match           = 1-np.abs(aae.autoencode(data)-data)[:,N:]
        performance["mae"][track]           = float(np.mean(data_match))                 # average bitwise match
        performance["prob_bitwise"][track]  = float(np.prod(np.mean(data_match,axis=0))) # product of bitwise match probabilty
        performance["prob_allmatch"][track] = float(np.mean(np.prod(data_match,axis=1))) # probability of complete match

    metrics(val,"val")
    metrics(train,"train")
    metrics(all_actions,"all")

    performance["effective_labels"] = int(effective_labels)

    import json
    with open(aae.local("performance.json"),"w") as f:
        json.dump(performance, f)

if "dump" in mode:
    # one-hot to id
    actions_byid = (actions * np.arange(num_actions)).sum(axis=-1,dtype=int)
    with open(sae.local("action_ids.csv"), 'wb') as f:
        np.savetxt(f,actions_byid,"%d")


"""* Summary:
Input: a subset of valid action pairs.

* Training:

* Evaluation:



If the number of actions are too large, they simply does not appear in the
training examples. This means those actions can be pruned, and you can lower the number of actions.


TODO:
verify all valid successors are generated, negative prior exploiting that fact

consider changing the input data: all successors are provided, closed world assumption

mearging action discriminator and state discriminator into one network


AD: use the minimum activation among the correct actions as a threshold
or use 1.0

AD: use action label as an additional input to discriminaotr (??)

AD: ensemble



"""

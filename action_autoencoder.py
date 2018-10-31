#!/usr/bin/env python3
import warnings
import config
import numpy as np
import latplan.model
from latplan.model import ActionAE
from latplan.util        import curry
from latplan.util.tuning import grid_search, nn_task

import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

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
}

if __name__ == '__main__':
    import numpy.random as random

    import sys
    if len(sys.argv) == 1:
        sys.exit("{} [directory]".format(sys.argv[0]))

    directory = sys.argv[1]
    directory_aae = "{}/_aae/".format(directory)
    mode = sys.argv[2]
    
    from latplan.util import get_ae_type
    ae = latplan.model.get(get_ae_type(directory))(directory).load()

    if "hanoi" in ae.path:
        data = np.loadtxt(ae.local("all_actions.csv"),dtype=np.int8)
    else:
        data = np.loadtxt(ae.local("actions.csv"),dtype=np.int8)
    
    parameters = {
        'N'          :[1],
        'M'          :[128],
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
    print(data.shape)
    try:
        if 'learn' in mode:
            raise Exception('learn')
        aae = ActionAE(directory_aae).load()
    except:
        aae,_,_ = grid_search(curry(nn_task, ActionAE, directory_aae,
                                    data[:int(len(data)*0.9)], data[:int(len(data)*0.9)],
                                    data[int(len(data)*0.9):], data[int(len(data)*0.9):],),
                              default_parameters,
                              parameters)
        aae.save()

    N = data.shape[1]//2
    
    actions = aae.encode_action(data, batch_size=1000).round()
    histogram = np.squeeze(actions.sum(axis=0,dtype=int))
    all_labels = np.zeros((np.count_nonzero(histogram), actions.shape[1], actions.shape[2]), dtype=int)
    for i, pos in enumerate(np.where(histogram > 0)[0]):
        all_labels[i][0][pos] = 1
    
    if 'plot' in mode:
        aae.plot(data[:8], "aae_train.png")
        aae.plot(data[int(len(data)*0.9):int(len(data)*0.9)+8], "aae_test.png")
        
        
        aae.plot(data[:8], "aae_train_decoded.png", ae=ae)
        aae.plot(data[int(len(data)*0.9):int(len(data)*0.9)+8], "aae_test_decoded.png", ae=ae)
        
        transitions = aae.decode([np.repeat(data[:1,:N], len(all_labels), axis=0), all_labels])
        aae.plot(transitions, "aae_all_actions_for_a_state.png", ae=ae)
        
        from latplan.util.timer import Timer
        # with Timer("loading csv..."):
        #     all_actions = np.loadtxt("{}/all_actions.csv".format(directory),dtype=np.int8)
        # transitions = aae.decode([np.repeat(all_actions[:1,:N], len(all_labels), axis=0), all_labels])
        suc = transitions[:,N:]
        from latplan.util.plot import plot_grid, squarify
        plot_grid([x for x in ae.decode_binary(suc)], w=8, path=aae.local("aae_all_actions_for_a_state_8x16.png"), verbose=True)
        plot_grid([x for x in ae.decode_binary(suc)], w=16, path=aae.local("aae_all_actions_for_a_state_16x8.png"), verbose=True)
        plot_grid(ae.decode_binary(data[:1,:N]), w=1, path=aae.local("aae_all_actions_for_a_state_state.png"), verbose=True)
        
    
    if 'check' in mode:
        from latplan.util.timer import Timer
        with Timer("loading csv..."):
            all_actions = np.loadtxt("{}/all_actions.csv".format(directory),dtype=np.int8)

        with Timer("shuffling"):
            random.shuffle(all_actions)
        all_actions = all_actions[:10000]

        count = 0
        try:
            pre_states = all_actions[:,:N]
            suc_states = all_actions[:,N:]
            pre_images = ae.decode_binary(pre_states,batch_size=1000)
            suc_images = ae.decode_binary(suc_states,batch_size=1000)

            import progressbar as pb
            bar = pb.ProgressBar(
                max_value=len(all_actions),
                widgets=[
                    pb.Timer("Elap: %(elapsed) "),
                    pb.AbsoluteETA("Est: %(elapsed) "),
                    pb.Bar(),
                ])
            for pre_state,suc_state,pre_image,suc_image in bar(zip(pre_states,suc_states,pre_images,suc_images)):
                
                generated_transitions = aae.decode([
                    np.repeat([pre_state],128,axis=0),
                    all_labels,
                ],batch_size=1000)
                generated_suc_states = generated_transitions[:,N:]
                generated_suc_images = ae.decode_binary(generated_suc_states,batch_size=1000)

                from latplan.util import bce
                errors = bce(generated_suc_images, np.repeat([suc_image],128,axis=0), axis=(1,2))
                min_error = np.amin(errors)
                if min_error < 0.01:
                    count += 1
        finally:
            print({"count": count, "total":len(all_actions)})
    
    actions = aae.encode_action(data, batch_size=1000)
    actions_r = actions.round()

    histogram = actions.sum(axis=0)
    print(histogram)
    histogram_r = actions_r.sum(axis=0,dtype=int)
    print(histogram_r)
    print (np.count_nonzero(histogram_r > 0))
        
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

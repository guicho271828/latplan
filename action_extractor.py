#!/usr/bin/env python3

import config
import numpy as np
from model import GumbelAE, ActionDiscriminator

import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.1f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

################################################################


from plot import plot_ae

def select(data,num):
    return data[random.randint(0,data.shape[0],num)]

def prepare(configs,ae):
    import numpy.random as random
    num = len(configs)
    transitions = mnist_puzzle.transitions(3,3,configs,True)
    pre = transitions[0]
    suc = transitions[1]
    # suc_invalid = np.copy(suc)
    # random.shuffle(suc_invalid)
    # transitions_invalid = np.concatenate((pre,suc_invalid),axis=1)
    pre_b = ae.encode_binary(pre).round()
    suc_b = ae.encode_binary(suc).round()
    suc_b_invalid = np.copy(suc_b)
    random.shuffle(suc_b_invalid)
    transition_b = np.concatenate((pre_b,suc_b),axis=1)
    invalid_transition_b = np.concatenate((pre_b,suc_b_invalid),axis=1)
    inputs = np.concatenate((transition_b,invalid_transition_b),axis=0)
    outputs = np.concatenate((np.ones(num),np.zeros(num)),axis=0)
    return inputs, outputs

def grid_search(path, epoch, train_in, train_out, test_in, test_out):
    names      = ['valid','invalid']
    parameters = [[100,1000],[1000,4000,],]
    best_error = float('inf')
    best_params = None
    best_ae     = None
    results = []
    try:
        import itertools
        import tensorflow as tf
        for params in itertools.product(*parameters):
            params_dict = { k:v for k,v in zip(names,params) }
            print("Testing model with parameters={}".format(params_dict))
            discriminator = ActionDiscriminator("samples/mnist_puzzle33p_ad/",params_dict)
            batch_size = 2000
            finished = False
            while not finished:
                print("batch size {}".format(batch_size))
                try:
                    discriminator.train(train_in, batch_size=batch_size, 
                                        test_data=test_in,
                                        train_data_to=train_out,
                                        test_data_to=test_out,
                                        epoch=epoch,)
                    finished = True
                except tf.errors.ResourceExhaustedError as e:
                    print(e)
                    batch_size = batch_size // 2
            error = discriminator.net.evaluate(test_in,test_out,batch_size=batch_size,)
            results.append((error,)+params)
            print("Evaluation result for {} : error = {}".format(params_dict,error))
            print("Current results:\n{}".format(np.array(results)),flush=True)
            if error < best_error:
                print("Found a better parameter {}: error:{} old-best:{}".format(
                    params_dict,error,best_error))
                best_params = params_dict
                best_error = error
                best_ae = discriminator
        print("Best parameter {}: error:{}".format(best_params,best_error))
    finally:
        print(results)
    best_ae.save()
    return best_ae,best_params,best_error


if __name__ == '__main__':
    import numpy.random as random
    from trace import trace
    import mnist_puzzle
    configs = mnist_puzzle.generate_configs(9)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)

    ae = GumbelAE("samples/mnist_puzzle33p_model/").load()

    train_n, test_n = 8000, 1000
    train_in, train_out = prepare(configs[:train_n],ae)
    test_in, test_out = prepare(configs[train_n:train_n+test_n],ae)

    train = True
    if train:
        discriminator, _, _ = grid_search("samples/mnist_puzzle33p_ad/",
                                    10, train_in, train_out, test_in, test_out)
        # discriminator = ActionDiscriminator("samples/mnist_puzzle33p_ad/",
        #                                     {'valid':1000,'invalid':2000})
        # discriminator.train(train_in, batch_size=1500, 
        #                     test_data=test_in,
        #                     train_data_to=train_out,
        #                     test_data_to=test_out,
        #                     # epoch=200,
        #                     # anneal_rate=0.0002,
        #                     epoch=1000,
        #                     anneal_rate=0.000008,)
    else:
        discriminator = ActionDiscriminator("samples/mnist_puzzle33p_ad/").load()
    print("index, discrimination, action")
    show_n = 10
    for i,a in enumerate(discriminator.discriminate(test_in)[:show_n]):
        print(i,a)
    for i,a in enumerate(discriminator.discriminate(test_in)[test_n:test_n+show_n]):
        print(i,a)
    
    
    

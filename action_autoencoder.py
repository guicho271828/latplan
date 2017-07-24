#!/usr/bin/env python3
import warnings
import config
import numpy as np
from latplan.model import ActionAE, default_networks

import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

################################################################

def curry(fn,*args1,**kwargs1):
    return lambda *args,**kwargs: fn(*args1,*args,**{**kwargs1,**kwargs})

def anneal_rate(epoch,min=0.1,max=5.0):
    import math
    return math.log(max/min) / epoch

# default values
default_parameters = {
    'lr'              : 0.0001,
    'batch_size'      : 2000,
    'full_epoch'      : 1000,
    'epoch'           : 1000,
    'max_temperature' : 5.0,
    'min_temperature' : 0.1,
}
parameters = {}

def learn_model(path,train_in,train_out,test_in,test_out,network,params_dict={}):
    discriminator = network(path)
    training_parameters = default_parameters.copy()
    for key, _ in training_parameters.items():
        if key in params_dict:
            training_parameters[key] = params_dict[key]
    discriminator.train(train_in,
                        test_data=test_in,
                        train_data_to=train_out,
                        test_data_to=test_out,
                        anneal_rate=anneal_rate(training_parameters['full_epoch'],
                                                training_parameters['min_temperature'],
                                                training_parameters['max_temperature']),
                        report=True,
                        **training_parameters,)
    return discriminator

def grid_search(path, train_in, train_out, test_in, test_out):
    # perform random trials on possible combinations
    network = ActionAE
    best_error = float('inf')
    best_params = None
    best_ae     = None
    results = []
    print("Network: {}".format(network))
    try:
        import itertools
        names  = [ k for k, _ in parameters.items()]
        values = [ v for _, v in parameters.items()]
        all_params = list(itertools.product(*values))
        random.shuffle(all_params)
        [ print(r) for r in all_params]
        for i,params in enumerate(all_params):
            config.reload_session()
            params_dict = { k:v for k,v in zip(names,params) }
            print("{}/{} Testing model with parameters=\n{}".format(i, len(all_params), params_dict))
            ae = learn_model(path, train_in,train_out,test_in,test_out,
                             network=curry(network, parameters=params_dict),
                             params_dict=params_dict)
            error = ae.net.evaluate(test_in,test_out,batch_size=100,verbose=0)
            results.append({'error':error, **params_dict})
            print("Evaluation result for:\n{}\nerror = {}".format(params_dict,error))
            print("Current results:")
            results.sort(key=lambda result: result['error'])
            [ print(r) for r in results]
            if error < best_error:
                print("Found a better parameter:\n{}\nerror:{} old-best:{}".format(
                    params_dict,error,best_error))
                del best_ae
                best_params = params_dict
                best_error = error
                best_ae = ae
                ae.plot(train_in[:8],"aae_train.png")
                ae.plot(test_in[:8],"aae_test.png")
                ae.save()
            else:
                del ae
        print("Best parameter:\n{}\nerror: {}".format(best_params,best_error))
    finally:
        print(results)
    with open(best_ae.local("grid_search.log"), 'a') as f:
        import json
        f.write("\n")
        json.dump(results, f)
    return best_ae,best_params,best_error


if __name__ == '__main__':
    import numpy.random as random

    import sys
    if len(sys.argv) == 1:
        sys.exit("{} [directory]".format(sys.argv[0]))

    directory = sys.argv[1]
    directory_aae = "{}/_aae/".format(directory)
    
    data = np.loadtxt("{}/actions.csv".format(directory),dtype=np.int8)
    
    global parameters
    parameters = {
        'N'          :[1],
        'M'          :[128],
        'layer'      :[1000],# [400,4000],
        'dropout'    :[0.4], #[0.1,0.4],
        'dropout_z'  :[False],
        'batch_size' :[2000],
        'full_epoch' :[500],
        'activation' :['tanh'],
        # quick eval
        'epoch'      :[500],
        'lr'         :[0.001],
    }
    print(data.shape)
    try:
        aae = ActionAE(directory_aae).load()
    except FileNotFoundError:
        aae,_,_ = grid_search(directory_aae, data[:12000], data[:12000], data[12000:], data[12000:],)

    aae.plot(data[:8], "aae_train.png")
    aae.plot(data[12000:12008], "aae_test.png")

    from latplan.util import get_ae_type
    ae = default_networks[get_ae_type(directory)](directory).load()

    aae.plot(data[:8], "aae_train_decoded.png", ae=ae)
    aae.plot(data[12000:12008], "aae_test_decoded.png", ae=ae)
    
    
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

"""

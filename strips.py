#!/usr/bin/env python3

import config
import numpy as np
from model import GumbelAE, ConvolutionalGumbelAE

float_formatter = lambda x: "%.5f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def curry(fn,*args1,**kwargs1):
    return lambda *args,**kwargs: fn(*args1,*args,**{**kwargs1,**kwargs})

def learn_model(path,train_data,test_data=None,network=GumbelAE):
    ae = network(path)
    ae.train(train_data,
             epoch=1000,
             anneal_rate=0.000008,
             # epoch=200,
             # anneal_rate=0.0002,
             max_temperature=5.0,
             # 
             batch_size=1000,
             test_data=test_data,
             min_temperature=0.1,
    )
    return ae

def grid_search(path, train=None, test=None , transitions=None, network=GumbelAE):
    parameters = [[2000],[0.4],]
    best_error = float('inf')
    best_params = None
    best_ae     = None
    results = []
    try:
        import itertools
        for params in itertools.product(*parameters):
            print("Testing model with parameters={}".format(params))
            ae = learn_model(path, train, test,
                             network=curry(network,parameters=params))
            error = ae.autoencoder.evaluate(test,test,batch_size=4000,)
            results.append((error,)+params)
            print("Evaluation result for {} : error = {}".format(params,error))
            print("Current results:\n{}".format(np.array(results)),flush=True)
            if error < best_error:
                print("Found a better parameter {}: error:{} old-best:{}".format(
                    params,error,best_error))
                best_params = params
                best_error = error
                best_ae = ae
        print("Best parameter {}: error:{}".format(best_params,best_error))
    finally:
        print(results)
    return best_ae,best_params,best_error

def dump_actions(ae,transitions):
    orig, dest = transitions[0], transitions[1]
    orig_b = ae.encode_binary(orig,batch_size=6000)
    dest_b = ae.encode_binary(dest,batch_size=6000)
    actions = np.concatenate((orig_b, dest_b), axis=1)
    print(ae.local("actions.csv"))
    np.savetxt(ae.local("actions.csv"),actions,"%d")

def dump(ae, path, train=None, test=None , transitions=None, **kwargs):
    if test is not None:
        plot_ae(ae,select(test,12),"autoencoding_test.png")
    plot_ae(ae,select(train,12),"autoencoding_train.png")
    if transitions is not None:
        dump_actions(ae,transitions)

################################################################

from plot import plot_ae

def select(data,num):
    return data[random.randint(0,data.shape[0],num)]

if __name__ == '__main__':
    import numpy.random as random
    import trace
    
    def run(*args, **kwargs):
        ae, _, _ = grid_search(*args, **kwargs)
        dump(ae, *args, **kwargs)
    
    # import counter
    # run("samples/counter_model/",
    #     counter.states(),
    #     None,
    #     counter.transitions(n=1000))
    # run("samples/counter_modelc/",
    #     counter.states(),
    #     None,
    #     counter.transitions(n=1000),
    #     network=ConvolutionalGumbelAE)
    # print( "################################################################")
    # import puzzle
    # run("samples/puzzle22_model/",
    #     puzzle.states(2,2).repeat(10,0),
    #     None,
    #     puzzle.transitions(2,2))
    # run("samples/puzzle22_modelc/",
    #     puzzle.states(2,2).repeat(10,0),
    #     None,
    #     puzzle.transitions(2,2),
    #     network=ConvolutionalGumbelAE)
    # print( "################################################################") 
    # import mnist_puzzle
    # run("samples/mnist_puzzle22_model/",
    #     mnist_puzzle.states(2,2).repeat(10,0),
    #     None,
    #     mnist_puzzle.transitions(2,2))
    # run("samples/mnist_puzzle22_modelc/",
    #     mnist_puzzle.states(2,2).repeat(10,0),
    #     None,
    #     mnist_puzzle.transitions(2,2),
    #     network=ConvolutionalGumbelAE) 
    # print( "################################################################") 
    # import puzzle
    # run("samples/puzzle32_model/",
    #     puzzle.states(3,2).repeat(10,0),
    #     None,
    #     puzzle.transitions(3,2))
    # run("samples/puzzle32_modelc/",
    #     puzzle.states(3,2).repeat(10,0),
    #     None,
    #     puzzle.transitions(3,2),
    #     network=ConvolutionalGumbelAE)
    
    # import puzzle
    # all_states = puzzle.states(3,2)
    # filter = random.choice([True, True, True, True, False, False, False,  False],
    #                        all_states.shape[0])
    # print(filter)
    # inv_filter = np.invert(filter)
    # run("samples/puzzle32p_model/",
    #     all_states[filter].repeat(10,0),
    #     all_states[inv_filter],
    #     puzzle.transitions(3,2))
    # ################################################################
    # import mnist_puzzle
    # run("samples/mnist_puzzle32_model/",
    #     mnist_puzzle.states(3,2).repeat(10,0),
    #     None,
    #     mnist_puzzle.transitions(3,2))
    # run("samples/mnist_puzzle32_modelc/",
    #     mnist_puzzle.states(3,2).repeat(10,0),
    #     None,
    #     mnist_puzzle.transitions(3,2),
    #     network=ConvolutionalGumbelAE) 

    import mnist_puzzle
    all_states = mnist_puzzle.states(3,2)
    filter = random.choice([True, False, False, False, False, False, False,  False],
                           all_states.shape[0])
    inv_filter = np.invert(filter)
    print(len(all_states),len(all_states[filter]),len(all_states[inv_filter]))
    run("samples/mnist_puzzle32p_model/",
        all_states[filter].repeat(40,0),
        all_states[inv_filter],
        mnist_puzzle.transitions(3,2))

# Dropout is useful for avoiding the overfitting, but requires larger epochs
# Too short epochs may result in underfitting

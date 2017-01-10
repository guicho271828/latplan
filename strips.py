#!/usr/bin/env python3

import numpy as np
from model import GumbelAE, ConvolutionalGumbelAE


def learn_model(path,train_data,test_data=None,network=GumbelAE):
    ae = network(path)
    ae.train(train_data,
             # epoch=500,
             # anneal_rate=0.00002,
             # max_temperature=1.0,
             epoch=200,
             anneal_rate=0.0002,
             max_temperature=5.0,
             # 
             batch_size=1000,
             test_data=test_data,
             min_temperature=0.1,
    )
    return ae

def dump_actions(ae,transitions):
    orig, dest = transitions[0], transitions[1]
    orig_b = ae.encode_binary(orig,batch_size=6000)
    dest_b = ae.encode_binary(dest,batch_size=6000)
    actions = np.concatenate((orig_b, dest_b), axis=1)
    np.savetxt(ae.local("actions.csv"),actions,"%d")

################################################################

from plot import plot_ae

def select(data,num):
    return data[random.randint(0,data.shape[0],num)]

if __name__ == '__main__':
    import numpy.random as random
    import trace

    def run(path, train_states, test_states=None , transitions=None, network=GumbelAE):
        ae = learn_model(path, train_states, test_states, network=network)
        if test_states is not None:
            plot_ae(ae,select(test_states,12),"autoencoding_test.png")
        plot_ae(ae,select(train_states,12),"autoencoding_train.png")
        if transitions is not None:
            dump_actions(ae,transitions)

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
    import mnist_puzzle
    run("samples/mnist_puzzle32_model/",
        mnist_puzzle.states(3,2).repeat(10,0),
        None,
        mnist_puzzle.transitions(3,2))
    # run("samples/mnist_puzzle32_modelc/",
    #     mnist_puzzle.states(3,2).repeat(10,0),
    #     None,
    #     mnist_puzzle.transitions(3,2),
    #     network=ConvolutionalGumbelAE) 

    import mnist_puzzle
    all_states = mnist_puzzle.states(3,2)
    filter = random.choice([True, True, True, True, False, False, False,  False],
                           all_states.shape[0])
    print(filter)
    inv_filter = np.invert(filter)
    run("samples/mnist_puzzle32p_model/",
        all_states[filter].repeat(20,0),
        all_states[inv_filter],
        mnist_puzzle.transitions(3,2))


#!/usr/bin/env python3

import config
import numpy as np
from model import GumbelAE, ConvolutionalGumbelAE

import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.5f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def curry(fn,*args1,**kwargs1):
    return lambda *args,**kwargs: fn(*args1,*args,**{**kwargs1,**kwargs})

def learn_model(path,train_data,test_data=None,network=GumbelAE):
    ae = network(path)
    ae.train(train_data,
             epoch=1000,
             anneal_rate=0.000008,
             # epoch=500,
             # anneal_rate=0.0001,
             # epoch=200,
             # anneal_rate=0.0002,
             max_temperature=5.0,
             # 
             batch_size=4000,
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
                             network=curry(network,
                                           parameters={i:v for i,v in enumerate(params)}))
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

# all_bitarray = np.unpackbits(np.arange(2**8, dtype=np.uint8).reshape((2**8,1)),axis=1)
# 
# def binary_counter(bitnum):           # 25
#     if bitnum > 8:              # yes
#         nextbitnum = bitnum - 8 # 17
#         for i in range(2**8):
#             for lowerbits in binary_counter(nextbitnum):
#                 yield np.concatenate((all_bitarray[i],lowerbits),axis=0)
#     else:
#         bitarray = all_bitarray[0:2**bitnum,8-bitnum:8]
#         for v in bitarray:
#             yield v

def flip(bv1,bv2):
    "bv1,bv2: integer 1D vector, whose values are 0 or 1"
    iv1 = np.packbits(bv1,axis=-1)
    iv2 = np.packbits(bv2,axis=-1)
    # print(iv1,iv2)
    # print(np.bitwise_xor(iv1,iv2))
    # print(np.unpackbits(np.bitwise_xor(iv1,iv2),axis=-1))
    # print(np.unpackbits(np.bitwise_xor(iv1,iv2),axis=-1).shape)
    # print(bv1.shape)
    return \
        np.unpackbits(np.bitwise_xor(iv1,iv2),axis=-1)[:, :bv1.shape[-1]]

def flips(bitnum,diffbit):
    # array = np.zeros(bitnum)
    def rec(start,diffbit,array):
        if diffbit > 0:
            for i in range(start,bitnum):
                this_array = np.copy(array)
                this_array[i] = 1
                for result in rec(i+1,diffbit-1,this_array):
                    yield result
        else:
            yield array
    return rec(0,diffbit,np.zeros(bitnum,dtype=np.int8))

def all_flips(bitnum,diffbit):
    size=1
    for i in range(bitnum-diffbit+1,bitnum+1):
        size *= i
    for i in range(1,diffbit+1):
        size /= i
    size = int(size)
    # print(size)
    array = np.zeros((size,bitnum),dtype=np.int8)
    import itertools
    for i,indices in enumerate(itertools.combinations(range(bitnum), diffbit)):
        array[i,indices] = 1
    return array

def augment_neighbors(ae, distance, bs1, bs2, threshold=0.,max_diff=None):
    bs1 = bs1.astype(np.int8)
    ys1 = ae.decode_binary(bs1,batch_size=6000)
    data_dim = np.prod(ys1.shape[1:])
    print("threshold {} corresponds to val_loss {}".format(threshold,threshold*data_dim))
    bitnum = bs1.shape[1]
    if max_diff is None:
        max_diff = bitnum-1
    final_bs1 = [bs1]
    final_bs2 = [bs2]
    failed_bv = []

    K.set_learning_phase(0)
    y_orig = K.placeholder(shape=ys1.shape)
    b = K.placeholder(shape=bs1.shape)
    z = tf.stack([b,1-b],axis=-1)
    y_flip = ae.decoder(z)
    ok = K.lesser_equal(distance(y_orig,y_flip),threshold)
    checker = K.function([y_orig,b],[ok])
    def check_ok(flipped_bs):
        return checker([ys1,flipped_bs])[0]

    last_skips = 0
    for diffbit in range(1,max_diff):
        some = False
        for bv in flips(bitnum,diffbit):
            if np.any([ np.all(np.greater_equal(bv,bv2)) for bv2 in failed_bv ]):
                # print("previously seen with failure")
                last_skips += 1
                continue
            print(bv, {"blk": len(failed_bv), "skip":last_skips, "acc":len(final_bs1)})
            last_skips = 0
            flipped_bs = flip(bs1,[bv])
            oks = check_ok(flipped_bs)
            new_bs = flipped_bs[oks]
            ok_num = len(new_bs)
            if ok_num > 0:
                some = True
                final_bs1.append(new_bs)
                # we do not enumerate destination states.
                # because various states are applicable, single destination state is enough
                final_bs2.append(bs2[oks])
            else:
                failed_bv.append(bv)
        if not some:
            print("No more augmentation, stopped")
            break
    return np.concatenate(final_bs1,axis=0), np.concatenate(final_bs2,axis=0)

def bce(x,y):
    return K.mean(K.binary_crossentropy(x,y),axis=(1,2))

# def bce(x,y):
#     x_sym = K.placeholder(shape=x.shape)
#     y_sym = K.placeholder(shape=y.shape)
#     diff_sym = K.mean(K.binary_crossentropy(x_sym,y_sym),axis=(1,2))
#     return K.function([x_sym,y_sym],[diff_sym])([x,y])[0]

def dump_actions(ae,transitions,threshold=0.):
    orig, dest = transitions[0], transitions[1]
    orig_b = ae.encode_binary(orig,batch_size=6000).round().astype(int)
    dest_b = ae.encode_binary(dest,batch_size=6000).round().astype(int)
    actions = np.concatenate((orig_b,dest_b), axis=1)
    print(ae.local("actions.csv"))
    np.savetxt(ae.local("actions.csv"),actions,"%d")
    actions = np.concatenate(
        augment_neighbors(ae,bce,orig_b,dest_b,threshold=0.09), axis=1)
    print(ae.local("augmented.csv"))
    np.savetxt(ae.local("augmented.csv"),actions,"%d")

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
    from trace import trace
    def run(learn,*args, **kwargs):
        if learn:
            ae, _, _ = grid_search(*args, **kwargs)
        else:
            ae = (lambda network=GumbelAE,**kwargs:network)(**kwargs)(args[0]).load()
            # ==network(path)
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

    # import mnist_puzzle
    # all_states = mnist_puzzle.states(3,2)
    # filter = random.choice([True, False, False, False, False, False, False,  False],
    #                        all_states.shape[0])
    # inv_filter = np.invert(filter)
    # print(len(all_states),len(all_states[filter]),len(all_states[inv_filter]))
    # run("samples/mnist_puzzle32p_model/",
    #     all_states[filter].repeat(40,0),
    #     all_states[inv_filter],
    #     mnist_puzzle.transitions(3,2))

    import mnist_puzzle
    configs = mnist_puzzle.generate_configs(9)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    train       = mnist_puzzle.states(3,3,train_c)
    test        = mnist_puzzle.states(3,3,test_c)
    transitions = mnist_puzzle.transitions(3,3,train_c)
    print(len(configs),len(train),len(test))
    run(True,"samples/mnist_puzzle33p_model/", train, test, transitions)

# Dropout is useful for avoiding the overfitting, but requires larger epochs
# Too short epochs may result in underfitting

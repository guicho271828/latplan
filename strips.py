#!/usr/bin/env python3

import config
import numpy as np
import numpy.random as random
from model import GumbelAE, ConvolutionalGumbelAE

import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.5f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def curry(fn,*args1,**kwargs1):
    return lambda *args,**kwargs: fn(*args1,*args,**{**kwargs1,**kwargs})

def anneal_rate(epoch,min=0.1,max=5.0):
    import math
    return (2 / (epoch * (epoch+1))) * math.log(max/min)

def learn_model(path,train_data,test_data=None,network=GumbelAE):
    ae = network(path)
    ae.train(train_data,
             epoch=10000,
             # epoch=500,
             # epoch=200,
             anneal_rate=anneal_rate(10000),
             batch_size=4000,
             test_data=test_data,
             report=False 
    )
    return ae

def grid_search(path, train=None, test=None, network=GumbelAE):
    names      = ['layer','dropout']
    parameters = [[4000],[0.4],]
    best_error = float('inf')
    best_params = None
    best_ae     = None
    results = []
    try:
        import itertools
        for params in itertools.product(*parameters):
            params_dict = { k:v for k,v in zip(names,params) }
            print("Testing model with parameters={}".format(params_dict))
            ae = learn_model(path, train, test,
                             network=curry(network, parameters=params_dict))
            error = ae.autoencoder.evaluate(test,test,batch_size=4000,verbose=0)
            results.append((error,)+params)
            print("Evaluation result for {} : error = {}".format(params_dict,error))
            print("Current results:\n{}".format(results),flush=True)
            if error < best_error:
                print("Found a better parameter {}: error:{} old-best:{}".format(
                    params_dict,error,best_error))
                best_params = params_dict
                best_error = error
                best_ae = ae
        print("Best parameter {}: error:{}".format(best_params,best_error))
    finally:
        print(results)
    best_ae.save()
    with open(best_ae.local("grid_search.log"), 'w') as f:
        import json
        json.dump(results, f)
    return best_ae,best_params,best_error

def flip(bv1,bv2):
    "bv1,bv2: integer 1D vector, whose values are 0 or 1"
    iv1 = np.packbits(bv1,axis=-1)
    iv2 = np.packbits(bv2,axis=-1)
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
    try:
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
    except KeyboardInterrupt:
        print("augmentation stopped")
    return np.concatenate(final_bs1,axis=0), np.concatenate(final_bs2,axis=0)

def bce(x,y):
    return K.mean(K.binary_crossentropy(x,y),axis=(1,2))

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

def dump(ae, train=None, test=None , transitions=None, **kwargs):
    if test is not None:
        plot_ae(ae,select(test,12),"autoencoding_test.png")
    plot_ae(ae,select(train,12),"autoencoding_train.png")
    if transitions is not None:
        dump_actions(ae,transitions)

def dump_all_actions(ae,configs,trans_fn):
    l = len(configs)
    batch = 10000
    loop = (l // batch) + 1
    try:
        with open(ae.local("all_actions.csv"), 'ab') as f:
            for begin in range(0,loop*batch,batch):
                end = begin + batch
                print((begin,end,len(configs)))
                transitions = trans_fn(configs[begin:end])
                orig, dest = transitions[0], transitions[1]
                orig_b = ae.encode_binary(orig,batch_size=6000).round().astype(int)
                dest_b = ae.encode_binary(dest,batch_size=6000).round().astype(int)
                actions = np.concatenate((orig_b,dest_b), axis=1)
                print(ae.local("all_actions.csv"))
                np.savetxt(f,actions,"%d")
    except KeyboardInterrupt:
        print("dump stopped")

################################################################

from plot import plot_ae

def select(data,num):
    return data[random.randint(0,data.shape[0],num)]

def run(learn,*args, **kwargs):
    if learn:
        ae, _, _ = grid_search(*args, **kwargs)
    else:
        ae = (lambda network=GumbelAE,**kwargs:network)(**kwargs)(args[0]).load()
    return ae

def run_mnist_puzzle():
    import puzzles.mnist_puzzle as p
    configs = p.generate_configs(9)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    train       = p.states(3,3,train_c)
    test        = p.states(3,3,test_c)
    print(len(configs),len(train),len(test))
    ae = run(True,"samples/mnist_puzzle33p_model/", train, test)
    dump(ae, train,test)
    dump_all_actions(ae,configs,lambda configs: p.transitions(3,3,configs))

def run_random_mnist_puzzle():
    import puzzles.random_mnist_puzzle as p
    configs = p.generate_configs(9)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    train       = p.states(3,3,train_c)
    test        = p.states(3,3,test_c)
    print(len(configs),len(train),len(test))
    ae = run(True,"samples/random_mnist_puzzle33p_model/", train, test)
    dump(ae, train,test)
    dump_all_actions(ae,configs,lambda configs: p.transitions(3,3,configs))

def run_lenna_puzzle():
    import puzzles.lenna_puzzle as p
    configs = p.generate_configs(9)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    train       = p.states(3,3,train_c)
    test        = p.states(3,3,test_c)
    print(len(configs),len(train),len(test))
    ae = run(True,"samples/lenna_puzzle33p_model_long/", train, test)
    dump(ae, train,test)
    dump_all_actions(ae,configs,lambda configs: p.transitions(3,3,configs))

def run_mandrill_puzzle():
    import puzzles.mandrill_puzzle as p
    configs = p.generate_configs(9)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    train       = p.states(3,3,train_c)
    test        = p.states(3,3,test_c)
    print(len(configs),len(train),len(test))
    ae = run(True,"samples/mandrill_puzzle33p_model/", train, test)
    dump(ae, train,test)
    dump_all_actions(ae,configs,lambda configs: p.transitions(3,3,configs))

def run_spider_puzzle():
    import puzzles.spider_puzzle as p
    configs = p.generate_configs(9)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    train       = p.states(3,3,train_c)
    test        = p.states(3,3,test_c)
    print(len(configs),len(train),len(test))
    ae = run(True,"samples/spider_puzzle33p_model_long/", train, test)
    dump(ae, train,test)
    dump_all_actions(ae,configs,lambda configs: p.transitions(3,3,configs))

def run_digital_puzzle():
    import puzzles.digital_puzzle as p
    configs = p.generate_configs(9)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    train       = p.states(3,3,train_c)
    test        = p.states(3,3,test_c)
    print(len(configs),len(train),len(test))
    ae = run(True,"samples/digital_puzzle33p_model/", train, test)
    dump(ae, train,test)
    dump_all_actions(ae,configs,lambda configs: p.transitions(3,3,configs))

def run_hanoi():
    import puzzles.hanoi as p
    configs = p.generate_configs(6)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    train       = p.states(6,train_c)
    test        = p.states(6,test_c)
    print(len(configs),len(train),len(test))
    ae = run(True,"samples/hanoi_model/", train, test)
    dump(ae, train,test)
    dump_all_actions(ae,configs,lambda configs: p.transitions(6,configs))

def run_digital_lightsout():
    import puzzles.digital_lightsout as p
    configs = np.repeat(p.generate_configs(3),1,axis=0)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:int(len(configs)*(0.8))]
    test_c  = configs[int(len(configs)*(0.8)):]
    print(train_c)
    train       = p.states(3,train_c)
    test        = p.states(3,test_c)
    print(len(configs),len(train),len(test))
    ae = run(True,"samples/digital_lightsout_model/", train, test)
    dump(ae, train,test)
    dump_all_actions(ae,configs,lambda configs: p.transitions(3,configs))

def run_mnist_counter():
    import puzzles.mnist_counter as p
    configs = np.repeat(p.generate_configs(10),10000,axis=0)
    states = p.states(10,configs)
    train       = states[:int(len(states)*(0.8))]
    test        = states[int(len(states)*(0.8)):]
    print(len(configs),len(train),len(test))
    ae = run(True,"samples/mnist_counter_model/", train, test)
    dump(ae, train,test)
    dump_all_actions(ae,configs,lambda configs: p.transitions(10,configs))


def run_random_mnist_counter():
    import puzzles.random_mnist_counter as p
    configs = np.repeat(p.generate_configs(10),10000,axis=0)
    states = p.states(10,configs)
    train       = states[:int(len(states)*(0.8))]
    test        = states[int(len(states)*(0.8)):]
    print(len(configs),len(train),len(test))
    ae = run(True,"samples/random_mnist_counter_model/", train, test)
    dump(ae, train,test)
    dump_all_actions(ae,configs,lambda configs: p.transitions(10,configs))

if __name__ == '__main__':
    from trace import trace
    # run_spider_puzzle()
    run_hanoi()
    # run_digital_lightsout()
    # try:
    #     run_digital_lightsout()
    # except:
    #     pass    
    # try:
    #     run_mnist_counter()
    # except:
    #     pass    
    # try:
    #     run_random_mnist_counter()
    # except:
    #     pass    
    # try:
    #     run_mnist_puzzle()
    # except:
    #     pass    
    # try:
    #     run_random_mnist_puzzle()
    # except:
    #     pass    
    

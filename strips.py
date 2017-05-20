#!/usr/bin/env python3

import config
import numpy as np
import numpy.random as random
from model import default_networks

import keras.backend as K
import tensorflow as tf


float_formatter = lambda x: "%.5f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def curry(fn,*args1,**kwargs1):
    return lambda *args,**kwargs: fn(*args1,*args,**{**kwargs1,**kwargs})

def anneal_rate(epoch,min=0.1,max=5.0):
    import math
    return math.log(max/min) / epoch

encoder = 'fc'
mode = 'learn_dump'
modes = {'learn':True,'learn_dump':True,'dump':False}
learn_flag = True

# default values
default_parameters = {
    'lr'              : 0.0001,
    'batch_size'      : 2000,
    'full_epoch'      : 1000,
    'epoch'           : 1000,
    'max_temperature' : 2.0,
    'min_temperature' : 0.1,
}
parameters = {}

def learn_model(path,train_data,test_data=None,network=None,params_dict={}):
    if network is None:
        network = default_networks[encoder]
    ae = network(path)
    training_parameters = default_parameters.copy()
    for key, _ in training_parameters.items():
        if key in params_dict:
            training_parameters[key] = params_dict[key]
    ae.train(train_data,
             anneal_rate=anneal_rate(training_parameters['full_epoch'],
                                     training_parameters['min_temperature'],
                                     training_parameters['max_temperature']),
             test_data=test_data,
             report=False,
             **training_parameters,)
    return ae

def grid_search(path, train=None, test=None):
    # perform random trials on possible combinations
    network = default_networks[encoder]
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
            ae = learn_model(path, train, test,
                             network=curry(network, parameters=params_dict),
                             params_dict=params_dict)
            error = ae.autoencoder.evaluate(test,test,batch_size=100,verbose=0)
            results.append({'error':error, **params_dict})
            print("Evaluation result for:\n{}\nerror = {}".format(params_dict,error))
            print("Current results:")
            results.sort(key=lambda result: result['error'])
            [ print(r) for r in results]
            if error < best_error:
                print("Found a better parameter:\n{}\nerror:{} old-best:{}".format(
                    params_dict,error,best_error))
                dump_autoencoding_image(ae,test,train)
                del best_ae
                best_params = params_dict
                best_error = error
                best_ae = ae
            else:
                del ae
        print("Best parameter:\n{}\nerror: {}".format(best_params,best_error))
    finally:
        print(results)
    best_ae.save()
    with open(best_ae.local("grid_search.log"), 'a') as f:
        import json
        f.write("\n")
        json.dump(results, f)
    return best_ae,best_params,best_error

# flip, ... dump_actions is currently unused
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
                print("No more augmentation, stopped\n")
                break
    except KeyboardInterrupt:
        print("augmentation stopped")
    return np.concatenate(final_bs1,axis=0), np.concatenate(final_bs2,axis=0)

def bce(x,y):
    return K.mean(K.binary_crossentropy(x,y),axis=(1,2))
        
def dump_actions(ae,transitions,threshold=0.,name="actions.csv"):
    orig, dest = transitions[0], transitions[1]
    orig_b = ae.encode_binary(orig,batch_size=6000).round().astype(int)
    dest_b = ae.encode_binary(dest,batch_size=6000).round().astype(int)
    actions = np.concatenate((orig_b,dest_b), axis=1)
    print(ae.local(name))
    np.savetxt(ae.local(name),actions,"%d")

    actions = np.concatenate(
        augment_neighbors(ae,bce,orig_b,dest_b,threshold=0.001), axis=1)
    print(ae.local("augmented.csv"))
    np.savetxt(ae.local("augmented.csv"),actions,"%d")

# This is used
def dump_autoencoding_image(ae,test,train):
    plot_ae(ae,select(test,12),"autoencoding_test.png")
    plot_ae(ae,select(train,12),"autoencoding_train.png")
    
def dump_all_actions(ae,configs,trans_fn,name="all_actions.csv",repeat=1):
    if 'dump' not in mode:
        return
    l = len(configs)
    batch = 10000
    loop = (l // batch) + 1
    try:
        print(ae.local(name))
        with open(ae.local(name), 'wb') as f:
            for i in range(repeat):
                for begin in range(0,loop*batch,batch):
                    end = begin + batch
                    print((begin,end,len(configs)))
                    transitions = trans_fn(configs[begin:end])
                    orig, dest = transitions[0], transitions[1]
                    orig_b = ae.encode_binary(orig,batch_size=1000).round().astype(int)
                    dest_b = ae.encode_binary(dest,batch_size=1000).round().astype(int)
                    actions = np.concatenate((orig_b,dest_b), axis=1)
                    np.savetxt(f,actions,"%d")
    except AttributeError:
        print("this AE does not support dumping")
    except KeyboardInterrupt:
        print("dump stopped")

def dump_all_states(ae,configs,states_fn,name="all_states.csv",repeat=1):
    if 'dump' not in mode:
        return
    l = len(configs)
    batch = 10000
    loop = (l // batch) + 1
    try:
        print(ae.local(name))
        with open(ae.local(name), 'wb') as f:
            for i in range(repeat):
                for begin in range(0,loop*batch,batch):
                    end = begin + batch
                    print((begin,end,len(configs)))
                    states = states_fn(configs[begin:end])
                    states_b = ae.encode_binary(states,batch_size=1000).round().astype(int)
                    np.savetxt(f,states_b,"%d")
    except AttributeError:
        print("this AE does not support dumping")
    except KeyboardInterrupt:
        print("dump stopped")

################################################################

# note: lightsout has epoch 200

from plot import plot_ae

def select(data,num):
    return data[random.randint(0,data.shape[0],num)]

def run(learn,*args, **kwargs):
    if learn:
        ae, _, _ = grid_search(*args, **kwargs)
    else:
        ae = default_networks[encoder](args[0]).load()
        ae.summary()
    return ae


def mnist_puzzle(width=3,height=3):
    global parameters
    parameters = {
        'layer'      :[2000],# [400,4000],
        'clayer'     :[16],# [400,4000],
        'dropout'    :[0.4], #[0.1,0.4],
        'N'          :[28],  #[25,49],
        'dropout_z'  :[False],
        'activation' : ['tanh'],
        'full_epoch' :[1000],
        'epoch'      :[500],
        'batch_size' :[2000],
        'lr'         :[0.001],
    }
    import puzzles.mnist_puzzle as p
    configs = p.generate_configs(width*height)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    train       = p.states(width,height,train_c)
    test        = p.states(width,height,test_c)
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/mnist_puzzle{}{}_{}/".format(width,height,encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs[:13000],lambda configs: p.transitions(width,height,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(width,height,configs),)
    dump_all_states(ae,configs[:13000],lambda configs: p.states(width,height,configs),"states.csv")
    dump_all_states(ae,configs,        lambda configs: p.states(width,height,configs),)

def random_mnist_puzzle(width=3,height=3):
    global parameters
    parameters = {
        'layer'      :[4000],
        'dropout'    :[0.4],
        'N'          :[49],
        'epoch'      :[1000],
        'batch_size' :[2000]
    }
    import puzzles.random_mnist_puzzle as p
    configs = p.generate_configs(width*height)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    train       = p.states(width,height,train_c)
    test        = p.states(width,height,test_c)
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/random_mnist_puzzle{}{}_{}/".format(width,height,encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs[:13000],lambda configs: p.transitions(width,height,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(width,height,configs),)
    
def lenna_puzzle(width=3,height=3):
    global parameters
    parameters = {
        'layer'      :[1000],# [400,4000],
        'clayer'     :[4],# [400,4000],
        'dropout'    :[0.1], #[0.1,0.4],
        'N'          :[36],  #[25,49],   # learn correctly with 36 bits; not with lower bits
        'dropout_z'  :[False],
        'activation' : ['tanh'],
        'full_epoch' :[1000],
        'epoch'      :[500],
        'batch_size' :[2000],
        'lr'         :[0.001],
    }
    import puzzles.lenna_puzzle as p
    configs = p.generate_configs(width*height)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    train       = p.states(width,height,train_c)
    test        = p.states(width,height,test_c)
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/lenna_puzzle{}{}_{}/".format(width,height,encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs[:13000],lambda configs: p.transitions(width,height,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(width,height,configs),)

def mandrill_puzzle(width=3,height=3):
    global parameters
    parameters = {
        'layer'      :[2000],# [400,4000],
        'clayer'     :[16],# [400,4000],
        'dropout'    :[0.4], #[0.1,0.4],
        'N'          :[28],  #[25,49],
        'dropout_z'  :[False],
        'activation' : ['tanh'],
        'full_epoch' :[1000],
        'epoch'      :[500],
        'batch_size' :[2000],
        'lr'         :[0.0001],
    }
    import puzzles.mandrill_puzzle as p
    configs = p.generate_configs(width*height)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    train       = p.states(width,height,train_c)
    test        = p.states(width,height,test_c)
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/mandrill_puzzle{}{}_{}/".format(width,height,encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs[:13000],lambda configs: p.transitions(width,height,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(width,height,configs),)

def hanoi_hard(disks=5):
    global parameters
    parameters = {
        # 'layer'      :[1000,2000,4000],
        # 'dropout'    :[0.1,0.2,0.4],
        # 'N'          :[16,25,30,36],
        # 'batch_size' :[100],
        # 'dropout_z'  :[True,False],
        # short results
        # {'dropout': 0.4 , 'dropout_z': False, 'N': 30, 'layer': 4000, 'error': -0.640392005443573}
        # {'dropout': 0.1 , 'dropout_z': True, 'N': 30, 'layer': 4000, 'error': -0.63293421268463135}
        # {'dropout': 0.2 , 'dropout_z': False, 'N': 30, 'layer': 2000, 'error': -0.62783586978912354}
        # {'dropout': 0.4 , 'dropout_z': False, 'N': 36, 'layer': 4000, 'error': -0.62599468231201172}
        # {'dropout': 0.1 , 'dropout_z': True, 'N': 30, 'layer': 2000, 'error': -0.62438416481018066}
        # {'dropout': 0.1 , 'dropout_z': True, 'N': 36, 'layer': 2000, 'error': -0.62357735633850098}
        # {'dropout': 0.1 , 'dropout_z': False, 'N': 30, 'layer': 2000, 'error': -0.61835479736328125}
        # {'dropout': 0.2 , 'dropout_z': True, 'N': 36, 'layer': 2000, 'error': -0.61645066738128662}
        # {'dropout': 0.2 , 'dropout_z': True, 'N': 36, 'layer': 4000, 'error': -0.61477929353713989}
        # {'dropout': 0.2 , 'dropout_z': True, 'N': 25, 'layer': 4000, 'error': -0.60899567604064941}
        # {'dropout': 0.4 , 'dropout_z': False, 'N': 25, 'layer': 4000, 'error': -0.60734283924102783}
        # {'dropout': 0.2 , 'dropout_z': False, 'N': 36, 'layer': 2000, 'error': -0.60594701766967773}
        # {'dropout': 0.2 , 'dropout_z': True, 'N': 30, 'layer': 4000, 'error': -0.60450851917266846}
        # {'dropout': 0.1 , 'dropout_z': True, 'N': 36, 'layer': 4000, 'error': -0.60306370258331299}
        # {'dropout': 0.1 , 'dropout_z': False, 'N': 36, 'layer': 4000, 'error': -0.60291892290115356}
        # {'dropout': 0.1 , 'dropout_z': False, 'N': 36, 'layer': 2000, 'error': -0.59930217266082764}
        # {'dropout': 0.2 , 'dropout_z': True, 'N': 25, 'layer': 2000, 'error': -0.59898924827575684}
        # {'dropout': 0.4 , 'dropout_z': True, 'N': 30, 'layer': 4000, 'error': -0.5976942777633667}
        # {'dropout': 0.4 , 'dropout_z': False, 'N': 30, 'layer': 2000, 'error': -0.59667587280273438}
        # {'dropout': 0.1 , 'dropout_z': True, 'N': 25, 'layer': 4000, 'error': -0.59463036060333252}
        # {'dropout': 0.1 , 'dropout_z': True, 'N': 25, 'layer': 2000, 'error': -0.59247839450836182}
        # {'dropout': 0.2 , 'dropout_z': False, 'N': 30, 'layer': 4000, 'error': -0.59211128950119019}
        # {'dropout': 0.4 , 'dropout_z': True, 'N': 36, 'layer': 4000, 'error': -0.59196054935455322}
        # {'dropout': 0.1 , 'dropout_z': False, 'N': 30, 'layer': 4000, 'error': -0.59061408042907715}
        # {'dropout': 0.2 , 'dropout_z': True, 'N': 30, 'layer': 2000, 'error': -0.59042090177536011}
        # {'dropout': 0.1 , 'dropout_z': False, 'N': 16, 'layer': 4000, 'error': -0.58864903450012207}
        # {'dropout': 0.4 , 'dropout_z': False, 'N': 25, 'layer': 2000, 'error': -0.58647298812866211}
        # {'dropout': 0.2 , 'dropout_z': False, 'N': 25, 'layer': 2000, 'error': -0.58424681425094604}
        # {'dropout': 0.1 , 'dropout_z': True, 'N': 36, 'layer': 1000, 'error': -0.58421194553375244}
        # {'dropout': 0.1 , 'dropout_z': False, 'N': 30, 'layer': 1000, 'error': -0.58234208822250366}
        # {'dropout': 0.2 , 'dropout_z': False, 'N': 36, 'layer': 1000, 'error': -0.58037769794464111}
        # {'dropout': 0.2 , 'dropout_z': False, 'N': 36, 'layer': 4000, 'error': -0.57998448610305786}
        # {'dropout': 0.1 , 'dropout_z': True, 'N': 16, 'layer': 4000, 'error': -0.57961761951446533}
        # {'dropout': 0.4 , 'dropout_z': False, 'N': 36, 'layer': 2000, 'error': -0.57958716154098511}
        # {'dropout': 0.4 , 'dropout_z': False, 'N': 16, 'layer': 4000, 'error': -0.57866716384887695}
        # {'dropout': 0.2 , 'dropout_z': True, 'N': 16, 'layer': 2000, 'error': -0.57798409461975098}
        # {'dropout': 0.1 , 'dropout_z': False, 'N': 25, 'layer': 1000, 'error': -0.57733619213104248}
        # {'dropout': 0.4 , 'dropout_z': True, 'N': 36, 'layer': 2000, 'error': -0.5758746862411499}
        # {'dropout': 0.4 , 'dropout_z': False, 'N': 16, 'layer': 2000, 'error': -0.57536393404006958}
        # {'dropout': 0.1 , 'dropout_z': True, 'N': 30, 'layer': 1000, 'error': -0.57484549283981323}
        # {'dropout': 0.2 , 'dropout_z': False, 'N': 25, 'layer': 1000, 'error': -0.57483327388763428}
        # {'dropout': 0.1 , 'dropout_z': False, 'N': 25, 'layer': 2000, 'error': -0.5738484263420105}
        # {'dropout': 0.2 , 'dropout_z': False, 'N': 16, 'layer': 4000, 'error': -0.57321882247924805}
        # {'dropout': 0.1 , 'dropout_z': True, 'N': 16, 'layer': 1000, 'error': -0.57276612520217896}
        # {'dropout': 0.2 , 'dropout_z': True, 'N': 36, 'layer': 1000, 'error': -0.57263761758804321}
        # {'dropout': 0.1 , 'dropout_z': True, 'N': 25, 'layer': 1000, 'error': -0.57030832767486572}
        # {'dropout': 0.1 , 'dropout_z': True, 'N': 16, 'layer': 2000, 'error': -0.56961214542388916}
        # {'dropout': 0.4 , 'dropout_z': True, 'N': 25, 'layer': 4000, 'error': -0.56890702247619629}
        # {'dropout': 0.4 , 'dropout_z': False, 'N': 30, 'layer': 1000, 'error': -0.56395626068115234}
        # {'dropout': 0.1 , 'dropout_z': False, 'N': 16, 'layer': 1000, 'error': -0.56385207176208496}
        # {'dropout': 0.2 , 'dropout_z': False, 'N': 25, 'layer': 4000, 'error': -0.562419593334198}
        # {'dropout': 0.4 , 'dropout_z': False, 'N': 36, 'layer': 1000, 'error': -0.55937016010284424}
        # {'dropout': 0.2 , 'dropout_z': True, 'N': 25, 'layer': 1000, 'error': -0.55886197090148926}
        # {'dropout': 0.4 , 'dropout_z': True, 'N': 25, 'layer': 2000, 'error': -0.55770504474639893}
        # {'dropout': 0.4 , 'dropout_z': True, 'N': 16, 'layer': 4000, 'error': -0.55719602108001709}
        # {'dropout': 0.2 , 'dropout_z': False, 'N': 16, 'layer': 2000, 'error': -0.55505228042602539}
        # {'dropout': 0.4 , 'dropout_z': False, 'N': 16, 'layer': 1000, 'error': -0.55492079257965088}
        # {'dropout': 0.1 , 'dropout_z': False, 'N': 36, 'layer': 1000, 'error': -0.5539441704750061}
        # {'dropout': 0.2 , 'dropout_z': False, 'N': 16, 'layer': 1000, 'error': -0.55313944816589355}
        # {'dropout': 0.1 , 'dropout_z': False, 'N': 16, 'layer': 2000, 'error': -0.55175542831420898}
        # {'dropout': 0.4 , 'dropout_z': False, 'N': 25, 'layer': 1000, 'error': -0.55133038759231567}
        # {'dropout': 0.2 , 'dropout_z': True, 'N': 16, 'layer': 4000, 'error': -0.54752600193023682}
        # {'dropout': 0.4 , 'dropout_z': True, 'N': 30, 'layer': 2000, 'error': -0.54715025424957275}
        # {'dropout': 0.2 , 'dropout_z': True, 'N': 30, 'layer': 1000, 'error': -0.54615998268127441}
        # {'dropout': 0.2 , 'dropout_z': False, 'N': 30, 'layer': 1000, 'error': -0.54047298431396484}
        # {'dropout': 0.4 , 'dropout_z': True, 'N': 16, 'layer': 1000, 'error': -0.53436446189880371}
        # {'dropout': 0.2 , 'dropout_z': True, 'N': 16, 'layer': 1000, 'error': -0.53258365392684937}
        # {'dropout': 0.4 , 'dropout_z': True, 'N': 30, 'layer': 1000, 'error': -0.53125882148742676}
        # {'dropout': 0.4 , 'dropout_z': True, 'N': 16, 'layer': 2000, 'error': -0.5290108323097229}
        # {'dropout': 0.4 , 'dropout_z': True, 'N': 36, 'layer': 1000, 'error': -0.5274394154548645}
        # {'dropout': 0.4 , 'dropout_z': True, 'N': 25, 'layer': 1000, 'error': -0.51308572292327881}
        # {'dropout': 0.1 , 'dropout_z': False, 'N': 25, 'layer': 4000, 'error': -0.50917017459869385}

        # pruned bad cases
        # 'layer'      :[2000,4000],
        # 'dropout'    :[0.2,0.4],
        # 'N'          :[25,30,36],
        # 'batch_size' :[100],
        # 'dropout_z'  :[True,False],
        # middle results (500/1000 epoch)
        # {'dropout': 0.4, 'error': -0.60835909843444824, 'layer': 4000, 'N': 25,'dropout_z': False}
        # {'dropout': 0.2, 'error': -0.6075860857963562,  'layer': 2000, 'N': 36,'dropout_z': True}
        # {'dropout': 0.4, 'error': -0.60412561893463135, 'layer': 4000, 'N': 36,'dropout_z': True}
        # {'dropout': 0.2, 'error': -0.59957921504974365, 'layer': 2000, 'N': 30,'dropout_z': True}
        # {'dropout': 0.2, 'error': -0.59358477592468262, 'layer': 2000, 'N': 30,'dropout_z': False}
        # {'dropout': 0.2, 'error': -0.59129548072814941, 'layer': 4000, 'N': 25,'dropout_z': False}
        # {'dropout': 0.4, 'error': -0.58963090181350708, 'layer': 2000, 'N': 30,'dropout_z': False}
        # {'dropout': 0.4, 'error': -0.58892679214477539, 'layer': 4000, 'N': 30,'dropout_z': True}
        # {'dropout': 0.2, 'error': -0.58624380826950073, 'layer': 4000, 'N': 36,'dropout_z': False}
        # {'dropout': 0.2, 'error': -0.58586376905441284, 'layer': 4000, 'N': 30,'dropout_z': True}
        # {'dropout': 0.2, 'error': -0.58075928688049316, 'layer': 2000, 'N': 25,'dropout_z': True}
        # {'dropout': 0.2, 'error': -0.57950294017791748, 'layer': 4000, 'N': 25,'dropout_z': True}
        # {'dropout': 0.2, 'error': -0.57386302947998047, 'layer': 4000, 'N': 30,'dropout_z': False}
        # {'dropout': 0.4, 'error': -0.57242345809936523, 'layer': 2000, 'N': 36,'dropout_z': False}
        # {'dropout': 0.4, 'error': -0.55973660945892334, 'layer': 4000, 'N': 25,'dropout_z': True}
        # {'dropout': 0.4, 'error': -0.55722737312316895, 'layer': 4000, 'N': 36,'dropout_z': False}
        # {'dropout': 0.4, 'error': -0.55417418479919434, 'layer': 2000, 'N': 30,'dropout_z': True}
        # {'dropout': 0.4, 'error': -0.55290007591247559, 'layer': 4000, 'N': 30,'dropout_z': False}
        # {'dropout': 0.4, 'error': -0.55126595497131348, 'layer': 2000, 'N': 36,'dropout_z': True}
        # {'dropout': 0.2, 'error': -0.54389727115631104, 'layer': 2000, 'N': 25,'dropout_z': False}
        # {'dropout': 0.4, 'error': -0.51387894153594971, 'layer': 2000, 'N': 25,'dropout_z': False}
        # {'dropout': 0.4, 'error': -0.49873256683349609, 'layer': 2000, 'N': 25,'dropout_z': True}

        # 'layer'      :[4000], #[1000,2000,4000]
        # 'dropout'    :[0.4],  #[0.1,0.2,0.4]
        # 'N'          :[30],   #[16,25,30,36],
        # 'batch_size' :[100],
        # 'dropout_z'  :[False],#[True,False],

        'layer'      :[5000], #[1000,2000,4000]
        'dropout'    :[0.6],  #[0.1,0.2,0.4]
        'N'          :[36],   #[16,25,30,36],
        'batch_size' :[200],
        'dropout_z'  :[False],#[True,False],

        'full_epoch' :[1000],
        # quick eval
        'epoch'      :[500],
    }
    import puzzles.hanoi as p
    configs = p.generate_configs(disks)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    print(len(configs))
    train_c = configs[:int(0.9*len(configs))]
    test_c  = configs[int(0.9*len(configs)):]
    train       = p.states(disks,train_c)
    test        = p.states(disks,test_c)
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/hanoi{}_{}/".format(disks,encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs,lambda configs: p.transitions(disks,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(disks,configs),)
    dump_all_states(ae,configs,lambda configs: p.states(disks,configs),"states.csv")
    dump_all_states(ae,configs,        lambda configs: p.states(disks,configs),)

def hanoi(disks=4):
    global parameters
    parameters = {
        'layer'      :[1000,2000,4000],# [400,4000],
        'dropout'    :[0.1,0.2,0.4], #[0.1,0.4],
        'N'          :[16,25,30,36],  #[25,49],
        'batch_size' :[500,2000],
        'dropout_z'  :[True,False],
        'full_epoch' :[1000],
        # quick eval
        'epoch'      :[1000],
    }
    import puzzles.hanoi as p
    configs = p.generate_configs(disks)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    print(len(configs))
    train_c = configs
    test_c  = configs
    train       = p.states(disks,train_c)
    test        = p.states(disks,test_c)
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/hanoi{}_{}/".format(disks,encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs,        lambda configs: p.transitions(disks,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(disks,configs),)

def xhanoi(disks=4):
    global parameters
    parameters = {
        'layer'      :[6000],
        'dropout'    :[0.4],
        'N'          :[29],
        'epoch'      :[10000],
        'batch_size' :[3500]
    }
    import puzzles.hanoi as p
    configs = p.generate_configs(disks)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    print(len(configs))
    train_c = configs
    test_c  = configs
    train       = p.states(disks,train_c)
    test        = p.states(disks,test_c)
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/xhanoi{}_{}/".format(disks,encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs,        lambda configs: p.transitions(disks,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(disks,configs),)

def digital_lightsout(size=4):
    global parameters
    parameters = {
        'layer'      :[2000],# [400,4000],
        'clayer'     :[16],# [400,4000],
        'dropout'    :[0.4], #[0.1,0.4],
        'N'          :[28],  #[25,49],
        'dropout_z'  :[False],
        'activation' : ['tanh'],
        'full_epoch' :[1000],
        'epoch'      :[500],
        'batch_size' :[2000],
        'lr'         :[0.001],
    }
    import puzzles.digital_lightsout as p
    print('generating configs...')
    configs = p.generate_configs(size)
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    print('generating figures...')
    train       = p.states(size,train_c)
    test        = p.states(size,test_c)

    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/digital_lightsout_{}_{}/".format(size,encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs[:13000],lambda configs: p.transitions(size,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(size,configs),)

def digital_lightsout_skewed(size=3):
    global parameters
    parameters = {
        'layer'      :[4000],
        'dropout'    :[0.4],
        'N'          :[49],
        'epoch'      :[1000],
        'batch_size' :[2000]
    }
    import puzzles.digital_lightsout_skewed as p
    print('generating configs...')
    configs = p.generate_configs(size)
    random.shuffle(configs)
    train_c = configs[:int(len(configs)*0.9)]
    test_c  = configs[int(len(configs)*0.9):]
    print('generating figures...')
    train       = p.states(size,train_c)
    test        = p.states(size,test_c)
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/digital_lightsout_skewed_{}_{}/".format(size,encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs,        lambda configs: p.transitions(size,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(size,configs))

def mnist_counter():
    global parameters
    parameters = {
        'layer'      :[4000],
        'dropout'    :[0.4],
        'N'          :[36],
        'epoch'      :[1000],
        'batch_size' :[3500]
    }
    import puzzles.mnist_counter as p
    configs = np.repeat(p.generate_configs(10),10000,axis=0)
    states = p.states(10,configs)
    train       = states[:int(len(states)*(0.8))]
    test        = states[int(len(states)*(0.8)):]
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/mnist_counter_{}/".format(encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs,        lambda configs: p.transitions(10,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(10,configs))

def random_mnist_counter():
    global parameters
    parameters = {
        'layer'      :[4000],
        'dropout'    :[0.4],
        'N'          :[36],
        'epoch'      :[1000],
        'batch_size' :[3500]
    }
    import puzzles.random_mnist_counter as p
    configs = np.repeat(p.generate_configs(10),10000,axis=0)
    states = p.states(10,configs)
    train       = states[:int(len(states)*(0.8))]
    test        = states[int(len(states)*(0.8)):]
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/random_mnist_counter_{}/".format(encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs,        lambda configs: p.transitions(10,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(10,configs))

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        print({ k for k in default_networks})
        gs = globals()
        print({ k for k in gs if hasattr(gs[k], '__call__')})
        print({k for k in modes})
    else:
        print('args:',sys.argv)
        sys.argv.pop(0)
        encoder = sys.argv.pop(0)
        if encoder not in default_networks:
            raise ValueError("invalid encoder!: {}".format(encoder))
        task = sys.argv.pop(0)
        mode = sys.argv.pop(0)
        if mode not in modes:
            raise ValueError("invalid mode!: {}".format(mode))
        learn_flag = modes[mode]
        globals()[task](*map(eval,sys.argv))

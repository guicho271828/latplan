#!/usr/bin/env python3
import warnings
import config
import numpy as np
from latplan.model import default_networks
from latplan.util        import curry, set_difference, prepare_binary_classification_data
from latplan.util.tuning import grid_search, nn_task
import numpy.random as random
import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

################################################################

ae = None
oae = None
sd3 = None

inflation = 1

def generate_nop(data):
    dim = data.shape[1]//2
    pre, suc = data[:,:dim], data[:,dim:]
    pre = np.concatenate((pre, suc), axis=0)
    data_invalid = np.concatenate((pre,pre),axis=1)
    data_invalid = set_difference(data_invalid, data)
    return data_invalid

def permute_suc(data):
    dim = data.shape[1]//2
    pre, suc = data[:,:dim], data[:,dim:]
    suc_invalid = np.copy(suc)
    random.shuffle(suc_invalid)
    data_invalid = np.concatenate((pre,suc_invalid),axis=1)
    data_invalid = set_difference(data_invalid, data)
    return data_invalid

def generate_random_action(data, sae):
    # reconstructable, maybe invalid
    dim = data.shape[1]//2
    pre, suc = data[:,:dim], data[:,dim:]
    from state_discriminator3 import generate_random
    pre = np.concatenate((pre, suc), axis=0)
    suc = np.concatenate((generate_random(pre, sae),
                          generate_random(pre, sae)), axis=0)[:len(pre)]
    actions_invalid = np.concatenate((pre, suc), axis=1)
    actions_invalid = set_difference(actions_invalid, data)
    return actions_invalid

def generate_random_action2(data):
    # completely random strings
    return np.random.randint(0,2,data.shape,dtype=np.int8)


def repeat_over(array, repeats, axis=0):
    array = np.expand_dims(array, axis)
    array = np.repeat(array, repeats, axis)
    return np.reshape(array,(*array.shape[:axis],-1,*array.shape[axis+2:]))

def generate_oae_action(known_transisitons):
    actions = oae.encode_action(known_transisitons, batch_size=1000).round()
    histogram = np.squeeze(actions.sum(axis=0,dtype=int))
    available_actions = np.zeros((np.count_nonzero(histogram), actions.shape[1], actions.shape[2]), dtype=int)
    for i, pos in enumerate(np.where(histogram > 0)[0]):
        available_actions[i][0][pos] = 1

    N = known_transisitons.shape[1] // 2
    states = known_transisitons.reshape(-1, N)
    y = oae.decode([
        # s1,s2,s3,s1,s2,s3,....
        repeat_over(states, len(available_actions), axis=0),
        # a1,a1,a1,a2,a2,a2,....
        np.repeat(available_actions, len(states), axis=0),], batch_size=1000) \
           .round().astype(np.int8)

    y = set_difference(y, known_transisitons)
    random.shuffle(y)

    return y

def prepare(data):
    print("discriminate correct transitions and nop, suc-permutation, reconstructable, and random bits combined")
    print("**** does not discriminate OEA-generated states quite well, do not use ****")
    data_invalid = np.concatenate(
        tuple([generate_nop(data),
               *[ permute_suc(data) for i in range(inflation) ],
               *[ generate_random_action(data, ae) for i in range(inflation) ],
               *[ generate_random_action2(data) for i in range(inflation) ]
        ]), axis=0)

    data_valid   = np.repeat(data, len(data_invalid)//len(data), axis=0)

    return prepare_binary_classification_data(data_valid, data_invalid)


def prepare_oae_validated(known_transisitons):
    # This is a cheating, since we assume validation oracle
    print("generate many actions from states using OEA (at least one action for each state is correct)",
          "validate it with validators, then discriminate the correct vs wrong transitions.",
          sep="\n")
    print("**** CHEATING ****")
    N = known_transisitons.shape[1] // 2
    y = generate_oae_action(known_transisitons)

    import latplan.puzzles.puzzle_mnist as p
    p.setup()
    batch = 100000
    valid_suc = np.zeros(len(y),dtype=bool)
    for i in range(1+len(y)//batch):
        print(i,"/",len(y)//batch)
        suc_images = ae.decode_binary(y[batch*i:batch*(i+1),N:],batch_size=1000)
        valid_suc[batch*i:batch*(i+1)] = p.validate_states(suc_images, 3,3,verbose=False,batch_size=1000)
        # This state validation is just for reducing the later effort for validating transitions
    
    before_len = len(y)
    y = y[valid_suc]
    print("removing invalid successor states:",before_len,"->",len(y))

    answers = np.zeros(len(y),dtype=int)
    for i in range(1+len(y)//batch):
        print(i,"/",len(y)//batch)
        pre_images = ae.decode_binary(y[batch*i:batch*(i+1),:N],batch_size=1000)
        suc_images = ae.decode_binary(y[batch*i:batch*(i+1),N:],batch_size=1000)
        answers[batch*i:batch*(i+1)] = np.array(p.validate_transitions([pre_images, suc_images], 3,3,batch_size=1000)).astype(int)
    
    l = len(y)
    positive = np.count_nonzero(answers)
    print(positive,l-positive)

    y_positive = y[answers.astype(bool)]
    y_negative = y[(1-answers).astype(bool)]
    y_negative = y_negative[:len(y_positive)]
    
    return prepare_binary_classification_data(y_positive, y_negative)

def prepare_oae_PU(known_transisitons):
    print("discriminate correct transitions and other transitions generated by OAE")
    y = generate_oae_action(known_transisitons)
    # normalize
    y = y[:len(known_transisitons)]
    return prepare_binary_classification_data(known_transisitons, y)

def prepare_oae_PU2(known_transisitons):
    print("discriminate correct transitions and other transitions generated by OAE, filtered by state validator")
    print("**** CHEATING ****")
    N = known_transisitons.shape[1] // 2
    y = generate_oae_action(known_transisitons)
    import latplan.puzzles.puzzle_mnist as p
    p.setup()
    batch = 100000
    valid_suc = np.zeros(len(y),dtype=bool)
    for i in range(1+len(y)//batch):
        print(i,"/",len(y)//batch)
        suc_images = ae.decode_binary(y[batch*i:batch*(i+1),N:],batch_size=1000)
        valid_suc[batch*i:batch*(i+1)] = p.validate_states(suc_images, 3,3,verbose=False,batch_size=1000)
    
    before_len = len(y)
    y = y[valid_suc]
    print("removing invalid successor states:",before_len,"->",len(y))
    y = y[:len(known_transisitons)]
    # normalize
    return prepare_binary_classification_data(known_transisitons, y)

def prepare_oae_PU3(known_transisitons):
    print("discriminate the correct transitions and the other transitions generated by OAE,",
          " filtered by the learned state discriminator",
          sep="\n")
    N = known_transisitons.shape[1] // 2
    y = generate_oae_action(known_transisitons)
    from latplan.util import get_ae_type
    from latplan.model import combined_discriminate, combined_discriminate2
    
    if "conv" not in get_ae_type(ae.path):
        cae = default_networks['SimpleCAE'](ae.local("_cae/")).load()
        ind = np.where(np.squeeze(combined_discriminate(y[:,N:],ae,cae,sd3,batch_size=1000)) > 0.5)[0]
    else:
        ind = np.where(np.squeeze(combined_discriminate2(y[:,N:],ae,sd3,batch_size=1000)) > 0.5)[0]
    
    y = y[ind]
    # y = y[:len(known_transisitons)]
    # normalize
    return prepare_binary_classification_data(known_transisitons, y)

def prepare_oae_PU4(known_transisitons):
    print("Learn from pre + action label",
          "*** INCOMPATIBLE MODEL! ***",
          sep="\n")
    N = known_transisitons.shape[1] // 2
    
    y = generate_oae_action(known_transisitons)
    
    from latplan.util import get_ae_type
    from latplan.model import combined_discriminate, combined_discriminate2
    
    if "conv" not in get_ae_type(ae.path):
        cae = default_networks['SimpleCAE'](ae.local("_cae/")).load()
        ind = np.where(np.squeeze(combined_discriminate(y[:,N:],ae,cae,sd3,batch_size=1000)) > 0.5)[0]
    else:
        ind = np.where(np.squeeze(combined_discriminate2(y[:,N:],ae,sd3,batch_size=1000)) > 0.5)[0]
    
    y = y[ind]

    actions = oae.encode_action(known_transisitons, batch_size=1000).round()
    positive = np.concatenate((known_transisitons[:,:N], np.squeeze(actions)), axis=1)
    actions = oae.encode_action(y, batch_size=1000).round()
    negative = np.concatenate((y[:,:N], np.squeeze(actions)), axis=1)
    # random.shuffle(negative)
    # negative = negative[:len(positive)]
    # normalize
    return prepare_binary_classification_data(positive, negative)

def prepare_oae_PU5(known_transisitons):
    print("Learn from pre + suc + action label",
          "*** INCOMPATIBLE MODEL! ***",
          sep="\n")
    N = known_transisitons.shape[1] // 2
       
    y = generate_oae_action(known_transisitons)
    
    from latplan.util import get_ae_type
    from latplan.model import combined_discriminate, combined_discriminate2
    
    if "conv" not in get_ae_type(ae.path):
        cae = default_networks['SimpleCAE'](ae.local("_cae/")).load()
        ind = np.where(np.squeeze(combined_discriminate(y[:,N:],ae,cae,sd3,batch_size=1000)) > 0.5)[0]
    else:
        ind = np.where(np.squeeze(combined_discriminate2(y[:,N:],ae,sd3,batch_size=1000)) > 0.5)[0]
    
    y = y[ind]

    actions = oae.encode_action(known_transisitons, batch_size=1000).round()
    positive = np.concatenate((known_transisitons, np.squeeze(actions)), axis=1)
    actions = oae.encode_action(y, batch_size=1000).round()
    negative = np.concatenate((y, np.squeeze(actions)), axis=1)
    # random.shuffle(negative)
    # negative = negative[:len(positive)]
    # normalize
    return prepare_binary_classification_data(positive, negative)


# default values
default_parameters = {
    'lr'              : 0.0001,
    'batch_size'      : 2000,
    'full_epoch'      : 1000,
    'epoch'           : 1000,
    'max_temperature' : 2.0,
    'min_temperature' : 0.1,
    'M'               : 2,
    'min_grad'        : 0.0,
}

# exhaustive tuning
parameters = {
    'num_layers' :[1,2,3],
    'layer'      :[50,300,1000],# [400,4000],
    'dropout'    :[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],    #[0.1,0.4], #0.6,0.7,
    'batch_size' :[4000],
    'full_epoch' :[1000],
    'activation' :['tanh','relu'],
    # quick eval
    'epoch'      :[3000],
    'lr'         :[0.001],
}

# tuned results
parameters = {
    'num_layers' :[1],
    'layer'      :[300],# [400,4000],
    'dropout'    :[0.8],    #[0.1,0.4], #0.6,0.7,
    'batch_size' :[4000],
    'full_epoch' :[1000],
    'activation' :['relu'],
    # quick eval
    'epoch'      :[3000],
    'lr'         :[0.001],
}

def bce(x,y):
    from keras.layers import Input
    from keras.models import Model
    i = Input(shape=x.shape[1:])
    m = Model(i,i)
    m.compile(optimizer="adam", loss='binary_crossentropy')
    return m.evaluate(x,y,batch_size=1000,verbose=0)

def mae(x,y):
    from keras.layers import Input
    from keras.models import Model
    i = Input(shape=x.shape[1:])
    m = Model(i,i)
    m.compile(optimizer="adam", loss='mean_absolute_error')
    return m.evaluate(x,y,batch_size=1000,verbose=0)

def test_oae_generated(directory,discriminator):
    print("--- additional testing on OAE-generated actions")

    known_transisitons = np.loadtxt(ae.local("actions.csv"),dtype=np.int8)
    y = generate_oae_action(known_transisitons)
    N = known_transisitons.shape[1] // 2
    
    answers = np.zeros(len(y),dtype=int)
    import latplan.puzzles.puzzle_mnist as p
    p.setup()
    batch = 100000
    for i in range(1+len(y)//batch):
        print(i,"/",len(y)//batch)
        pre_images = ae.decode_binary(y[batch*i:batch*(i+1),:N],batch_size=1000)
        suc_images = ae.decode_binary(y[batch*i:batch*(i+1),N:],batch_size=1000)
        answers[batch*i:batch*(i+1)] = np.array(p.validate_transitions([pre_images, suc_images], 3,3,batch_size=1000)).astype(int)

    # discriminator.report(y, train_data_to=answers) # not appropriate for PUDiscriminator
    predictions = discriminator.discriminate(y,batch_size=1000)

    # print("BCE:", bce(predictions, answers))
    # print("accuracy:", 100-mae(predictions.round(), answers)*100, "%")

    from latplan.util import get_ae_type
    from latplan.model import combined_discriminate, combined_discriminate2

    if "conv" not in get_ae_type(directory):
        cae = default_networks['SimpleCAE'](ae.local("_cae/")).load()
        ind = np.where(np.squeeze(combined_discriminate(y[:,N:],ae,cae,sd3,batch_size=1000)) > 0.5)[0]
    else:
        ind = np.where(np.squeeze(combined_discriminate2(y[:,N:],ae,sd3,batch_size=1000)) > 0.5)[0]
    print("BCE (w/o invalid states by sd3):", bce(predictions[ind], answers[ind]))
    print("accuracy (w/o invalid states by sd3):", 100-mae(predictions[ind].round(), answers[ind])*100, "%")

    ind = p.validate_states(ae.decode_binary(y[:,N:],batch_size=1000),3,3,verbose=False,batch_size=1000)
    print("BCE (w/o invalid states by validator):", bce(predictions[ind], answers[ind]))
    print("accuracy (w/o invalid states by validator):", 100-mae(predictions[ind].round(), answers[ind])*100, "%")

def test_oae_pre_label(directory,discriminator):
    print("--- additional testing on OAE-generated actions")

    known_transisitons = np.loadtxt(ae.local("actions.csv"),dtype=np.int8)
    y = generate_oae_action(known_transisitons)
    N = known_transisitons.shape[1] // 2
    
    answers = np.zeros(len(y),dtype=int)
    import latplan.puzzles.puzzle_mnist as p
    p.setup()
    batch = 100000
    for i in range(1+len(y)//batch):
        print(i,"/",len(y)//batch)
        pre_images = ae.decode_binary(y[batch*i:batch*(i+1),:N],batch_size=1000)
        suc_images = ae.decode_binary(y[batch*i:batch*(i+1),N:],batch_size=1000)
        answers[batch*i:batch*(i+1)] = np.array(p.validate_transitions([pre_images, suc_images], 3,3,batch_size=1000)).astype(int)

    # discriminator.report(y, train_data_to=answers) # not appropriate for PUDiscriminator
    actions = oae.encode_action(y, batch_size=1000).round()
    predictions = discriminator.discriminate(np.concatenate((y[:,:N], np.squeeze(actions)), axis=1),batch_size=1000)

    # print("BCE:", bce(predictions, answers))
    # print("accuracy:", 100-mae(predictions.round(), answers)*100, "%")

    from latplan.util import get_ae_type
    from latplan.model import combined_discriminate, combined_discriminate2

    if "conv" not in get_ae_type(directory):
        cae = default_networks['SimpleCAE'](ae.local("_cae/")).load()
        ind = np.where(np.squeeze(combined_discriminate(y[:,N:],ae,cae,sd3,batch_size=1000)) > 0.5)[0]
    else:
        ind = np.where(np.squeeze(combined_discriminate2(y[:,N:],ae,sd3,batch_size=1000)) > 0.5)[0]
    print("BCE (w/o invalid states by sd3):", bce(predictions[ind], answers[ind]))
    print("accuracy (w/o invalid states by sd3):", 100-mae(predictions[ind].round(), answers[ind])*100, "%")

    ind = p.validate_states(ae.decode_binary(y[:,N:],batch_size=1000),3,3,verbose=False,batch_size=1000)
    print("BCE (w/o invalid states by validator):", bce(predictions[ind], answers[ind]))
    print("accuracy (w/o invalid states by validator):", 100-mae(predictions[ind].round(), answers[ind])*100, "%")

def test_oae_pre_suc_label(directory,discriminator):
    print("--- additional testing on OAE-generated actions")

    known_transisitons = np.loadtxt(ae.local("actions.csv"),dtype=np.int8)
    y = generate_oae_action(known_transisitons)
    N = known_transisitons.shape[1] // 2
    
    answers = np.zeros(len(y),dtype=int)
    import latplan.puzzles.puzzle_mnist as p
    p.setup()
    batch = 100000
    for i in range(1+len(y)//batch):
        print(i,"/",len(y)//batch)
        pre_images = ae.decode_binary(y[batch*i:batch*(i+1),:N],batch_size=1000)
        suc_images = ae.decode_binary(y[batch*i:batch*(i+1),N:],batch_size=1000)
        answers[batch*i:batch*(i+1)] = np.array(p.validate_transitions([pre_images, suc_images], 3,3,batch_size=1000)).astype(int)

    # discriminator.report(y, train_data_to=answers) # not appropriate for PUDiscriminator
    actions = oae.encode_action(y, batch_size=1000).round()
    predictions = discriminator.discriminate(np.concatenate((y, np.squeeze(actions)), axis=1),batch_size=1000)

    # print("BCE:", bce(predictions, answers))
    # print("accuracy:", 100-mae(predictions.round(), answers)*100, "%")

    from latplan.util import get_ae_type
    from latplan.model import combined_discriminate, combined_discriminate2

    if "conv" not in get_ae_type(directory):
        cae = default_networks['SimpleCAE'](ae.local("_cae/")).load()
        ind = np.where(np.squeeze(combined_discriminate(y[:,N:],ae,cae,sd3,batch_size=1000)) > 0.5)[0]
    else:
        ind = np.where(np.squeeze(combined_discriminate2(y[:,N:],ae,sd3,batch_size=1000)) > 0.5)[0]
    print("BCE (w/o invalid states by sd3):", bce(predictions[ind], answers[ind]))
    print("accuracy (w/o invalid states by sd3):", 100-mae(predictions[ind].round(), answers[ind])*100, "%")

    ind = p.validate_states(ae.decode_binary(y[:,N:],batch_size=1000),3,3,verbose=False,batch_size=1000)
    print("BCE (w/o invalid states by validator):", bce(predictions[ind], answers[ind]))
    print("accuracy (w/o invalid states by validator):", 100-mae(predictions[ind].round(), answers[ind])*100, "%")


def main(directory, mode, input_type=prepare_oae_PU3):
    directory_ad = "{}/_ad/".format(directory)
    print(directory, mode, input_type)
    from latplan.util import get_ae_type
    global ae, oae, sd3
    ae = default_networks[get_ae_type(directory)](directory).load()
    oae = default_networks['ActionAE'](ae.local("_aae/")).load()
    sd3 = default_networks['PUDiscriminator'](ae.local("_sd3/")).load()

    try:
        if 'learn' in mode:
            raise Exception('learn')
        if input_type is prepare_oae_validated:
            discriminator = default_networks['Discriminator'](directory_ad).load()
        else:
            discriminator = default_networks['PUDiscriminator'](directory_ad).load()
    except:
        data = np.loadtxt("{}/actions.csv".format(directory),dtype=np.int8)
        # data = np.loadtxt("{}/all_actions.csv".format(directory),dtype=np.int8)
        if input_type is prepare_oae_validated:
            train_in, train_out, test_in, test_out = prepare_oae_validated(data)
            discriminator,_,_ = grid_search(curry(nn_task, default_networks['Discriminator'], directory_ad,
                                                  train_in, train_out, test_in, test_out,),
                                            default_parameters,
                                            parameters)
        else:
            train_in, train_out, test_in, test_out = input_type(data)
            discriminator,_,_ = grid_search(curry(nn_task, default_networks['PUDiscriminator'], directory_ad,
                                                  train_in, train_out, test_in, test_out,),
                                            default_parameters,
                                            parameters)
        discriminator.save()

    if input_type is prepare_oae_PU4:
        test_oae_pre_label(directory,discriminator)
    elif input_type is prepare_oae_PU5:
        test_oae_pre_suc_label(directory,discriminator)
    else:
        test_oae_generated(directory,discriminator)

    # skipping the rest of the tests
    import sys
    sys.exit(0)
    
    # test if the learned action is correct

    # actions_valid = np.loadtxt("{}/actions.csv".format(directory),dtype=int)
    actions_valid = np.loadtxt("{}/all_actions.csv".format(directory),dtype=np.int8)
    random.shuffle(actions_valid)
    actions_valid = actions_valid[:100000]
    
    N = actions_valid.shape[1] // 2
    print("valid",actions_valid.shape)
    discriminator.report(actions_valid,  train_data_to=np.ones((len(actions_valid),)))
    print("type1 error: ",np.mean(1-np.round(discriminator.discriminate(actions_valid,batch_size=1000)))*100, "%")

    c = 0
    def type2(actions_invalid, message):
        nonlocal c
        c += 1
        actions_invalid = set_difference(actions_invalid, actions_valid)
        print("invalid",c,actions_invalid.shape, "---", message)
        discriminator.report(actions_invalid,train_data_to=np.zeros((len(actions_invalid),)))
        print("type2 error:",np.mean(np.round(discriminator.discriminate(actions_invalid,batch_size=1000))) * 100, "%")

        if 'check' in mode:
            import latplan.puzzles.puzzle_mnist as p
            p.setup()
            count = 0
            batch = 10000
            for i in range(len(actions_invalid)//batch):
                pre_images = ae.decode_binary(actions_invalid[batch*i:batch*(i+1),:N],batch_size=1000)
                suc_images = ae.decode_binary(actions_invalid[batch*i:batch*(i+1),N:],batch_size=1000)
                validation = p.validate_transitions([pre_images, suc_images], 3,3)
                count += np.count_nonzero(validation)
            print(count,"valid actions in invalid", c)

    
    type2(np.random.randint(0,2,(len(actions_valid),2*N),dtype=np.int8),
          "invalid actions generated from random bits (both pre and suc)")
        
    type2(generate_random_action(actions_valid, ae),
          "sucessors are random reconstructable states (incl. invalid states such as those with duplicated tiles)")
    
    pre, suc = actions_valid[:,:N], actions_valid[:,N:]
    suc_invalid = np.copy(suc)
    random.shuffle(suc_invalid)
    type2(np.concatenate((pre,suc_invalid),axis=1),
          "generated by swapping successors; pre/suc are correct states from the training examples")

    pre2 = np.loadtxt("{}/all_states.csv".format(directory),dtype=np.int8)
    suc2 = np.copy(pre2)
    random.shuffle(suc2)
    type2(np.concatenate((pre2,suc2),axis=1),
          "generated by shuffling all valid states; pre/suc are correct but possibly unknown states")
    
    type2(np.concatenate((pre2, pre2), axis=1),
          "invalid actions generated by nop")

    type2(np.concatenate((np.random.randint(0,2,(len(actions_valid),N),dtype=np.int8), suc), axis=1),
          "pre are generated by random bits, suc are correct states from the training examples")

    type2(np.concatenate((pre, np.random.randint(0,2,(len(actions_valid),N),dtype=np.int8)), axis=1),
          "suc are generated by random bits, pre are correct states from the training examples")
    
"""

* Summary:
Input: a subset of valid action pairs.

* Training:
From the valid action pairs, pseudo-invalid action pairs are generated by randomly swapping the successor states.
(graph sparsity assumption)

Oracle is trained to classify valid and pseudo-invalid action pairs. output of
the network is a single value indicating valid (1) and invalid (0).

The valid and the pseudo-invalid pairs are concatenated, randomly reordered and
divided into training samples and validation samples (for checking if
it is not overfitting. This does not by itself show the correctness of the
learned model)

* Evaluation:

Dataset: mnist_puzzle33_fc2

The result is validated on the entire valid action set (967680) and 2 sets of invalid action set.
Invalid action set (1) is created by removing the valid actions from a set of 10000 randomly generated bit vectors.
Invalid action set (2) is created by removing the valid actions from a set generated by swapping the successors of valid actions.
This guarantees all "states" are correct; thus the NN is not detecting the "unrealistic" states, and purely looking at the transitions.
Invalid action set (3): from invalid states to valid states.
Invalid action set (4): from valid states to invalid states.

type-1 error for the entire valid action set (967680 actions):
Mean Absolute Error: 

type-2 error for the invalid action set (1):
Mean Absolute Error: 

type-2 error for the invalid action set (2):
Mean Absolute Error: 

type-2 error for the invalid action set (3):
Mean Absolute Error: 

type-2 error for the invalid action set (4):
Mean Absolute Error: 


"""

if __name__ == '__main__':
    import sys
    def myeval(str):
        try:
            return eval(str)
        except:
            return str
    print(list(map(myeval,sys.argv[1:])))
    main(*map(myeval,sys.argv[1:]))

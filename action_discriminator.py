#!/usr/bin/env python3
import warnings
import config
import numpy as np
import latplan
import latplan.model
from latplan.util        import curry, set_difference, prepare_binary_classification_data
from latplan.util.tuning import grid_search, nn_task
import numpy.random as random
import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

################################################################

sae = None
oae = None
cae = None
sd3 = None
discriminator = None

def combined(states):
    from latplan.util import get_ae_type
    from latplan.model import combined_discriminate, combined_discriminate2
    if cae:
        return combined_discriminate(states,sae,cae,sd3,batch_size=1000)
    else:
        return combined_discriminate2(states,sae,sd3,batch_size=1000)

################################################################
# random action generators
    
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
    print("listing actions")
    actions = oae.encode_action(known_transisitons, batch_size=1000).round()
    histogram = np.squeeze(actions.sum(axis=0,dtype=int))
    available_actions = np.zeros((np.count_nonzero(histogram), actions.shape[1], actions.shape[2]), dtype=int)
    for i, pos in enumerate(np.where(histogram > 0)[0]):
        available_actions[i][0][pos] = 1

    N = known_transisitons.shape[1] // 2
    states = known_transisitons.reshape(-1, N)
    print("start generating transitions")
    y = oae.decode([
        # s1,s2,s3,s1,s2,s3,....
        repeat_over(states, len(available_actions), axis=0),
        # a1,a1,a1,a2,a2,a2,....
        np.repeat(available_actions, len(states), axis=0),], batch_size=1000) \
           .round().astype(np.int8)

    print("remove known transitions")
    y = set_difference(y, known_transisitons)
    print("shuffling")
    random.shuffle(y)
    return y

################################################################
# data preparation

def prepare(data):
    print("discriminate correct transitions and nop, suc-permutation, reconstructable, and random bits combined")
    print("**** does not discriminate OEA-generated states quite well, do not use ****")
    data_invalid = np.concatenate(
        tuple([generate_nop(data),
               *[ permute_suc(data) for i in range(inflation) ],
               *[ generate_random_action(data, sae) for i in range(inflation) ],
               *[ generate_random_action2(data) for i in range(inflation) ]
        ]), axis=0)

    data_valid   = np.repeat(data, len(data_invalid)//len(data), axis=0)

    return (latplan.model.get('PUDiscriminator'), *prepare_binary_classification_data(data_valid, data_invalid))

# This is a cheating, since we assume validation oracle
def prepare_oae_validated(known_transisitons):
    print("generate many actions from states using OEA (at least one action for each state is correct)",
          "validate it with validators, then discriminate the correct vs wrong transitions.",
          sep="\n")
    print("**** CHEATING ****")
    N = known_transisitons.shape[1] // 2
    y = generate_oae_action(known_transisitons)

    p = latplan.util.puzzle_module(sae.path)
    batch = 100000
    valid_suc = np.zeros(len(y),dtype=bool)
    for i in range(1+len(y)//batch):
        print(i,"/",len(y)//batch)
        suc_images = sae.decode_binary(y[batch*i:batch*(i+1),N:],batch_size=1000)
        valid_suc[batch*i:batch*(i+1)] = p.validate_states(suc_images,verbose=False,batch_size=1000)
        # This state validation is just for reducing the later effort for validating transitions
    
    before_len = len(y)
    y = y[valid_suc]
    print("removing invalid successor states:",before_len,"->",len(y))

    answers = np.zeros(len(y),dtype=int)
    for i in range(1+len(y)//batch):
        print(i,"/",len(y)//batch)
        pre_images = sae.decode_binary(y[batch*i:batch*(i+1),:N],batch_size=1000)
        suc_images = sae.decode_binary(y[batch*i:batch*(i+1),N:],batch_size=1000)
        answers[batch*i:batch*(i+1)] = np.array(p.validate_transitions([pre_images, suc_images],batch_size=1000)).astype(int)
    
    l = len(y)
    positive = np.count_nonzero(answers)
    print(positive,l-positive)

    y_positive = y[answers.astype(bool)]
    y_negative = y[(1-answers).astype(bool)]
    y_negative = y_negative[:len(y_positive)]
    
    return (latplan.model.get('Discriminator'), *prepare_binary_classification_data(y_positive, y_negative))

# discriminate correct transitions and other transitions generated by OAE
def prepare_oae_PU(known_transisitons):
    print("discriminate correct transitions and other transitions generated by OAE")
    y = generate_oae_action(known_transisitons)
    # normalize
    y = y[:len(known_transisitons)]
    return (latplan.model.get('PUDiscriminator'), *prepare_binary_classification_data(known_transisitons, y))

# discriminate the correct transitions and the other transitions generated by OAE,
# filtered by the state validator ***CHEATING***
def prepare_oae_PU2(known_transisitons):
    print("discriminate the correct transitions and the other transitions generated by OAE, filtered by the state validator")
    print("**** CHEATING ****")
    N = known_transisitons.shape[1] // 2
    y = generate_oae_action(known_transisitons)
    p = latplan.util.puzzle_module(sae.path)
    batch = 100000
    valid_suc = np.zeros(len(y),dtype=bool)
    for i in range(1+len(y)//batch):
        print(i,"/",len(y)//batch)
        suc_images = sae.decode_binary(y[batch*i:batch*(i+1),N:],batch_size=1000)
        valid_suc[batch*i:batch*(i+1)] = p.validate_states(suc_images,verbose=False,batch_size=1000)
    
    before_len = len(y)
    y = y[valid_suc]
    print("removing invalid successor states:",before_len,"->",len(y))
    y = y[:len(known_transisitons)]
    # normalize
    return (latplan.model.get('PUDiscriminator'), *prepare_binary_classification_data(known_transisitons, y))

# discriminate the correct transitions and the other transitions generated by OAE,
# filtered by the learned state discriminator
def prepare_oae_PU3(known_transisitons):
    print("discriminate the correct transitions and the other transitions generated by OAE,",
          " filtered by the learned state discriminator",
          sep="\n")
    N = known_transisitons.shape[1] // 2
    y = generate_oae_action(known_transisitons)

    print("removing invalid successors (sd3)")
    ind = np.where(np.squeeze(combined(y[:,N:])) > 0.5)[0]
    
    y = y[ind]
    if len(known_transisitons) > 100:
        y = y[:len(known_transisitons)] # undersample
    
    print("valid:",len(known_transisitons),"mixed:",len(y),)
    print("creating binary classification labels")
    return (latplan.model.get('PUDiscriminator'), *prepare_binary_classification_data(known_transisitons, y))

################################################################
# training parameters

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
    'batch_size' :[1000],
    'full_epoch' :[1000],
    'activation' :['tanh','relu'],
    # quick eval
    'epoch'      :[3000],
    'lr'         :[0.001],
}

# tuned results
parameters = {
    'num_layers' :[1,2],
    'layer'      :[300],# [400,4000],
    'dropout'    :[0.5, 0.8],    #[0.1,0.4], #0.6,0.7,
    'batch_size' :[1000],
    'full_epoch' :[1000],
    'activation' :['relu'],
    # quick eval
    'epoch'      :[3000],
    'lr'         :[0.001],
}

# good for puzzles
# {"dropout": 0.8, "full_epoch": 1000, "layer": 300, "num_layers": 1,
#  "batch_size": 1000, "activation": "relu", "epoch": 3000, "lr": 0.001}
# good for lightsout
# {'dropout': 0.5, 'full_epoch': 1000, 'layer': 300, 'num_layers': 2,
#  'batch_size': 1000, 'activation': 'relu', 'epoch': 3000, 'lr': 0.001}

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

def learn(input_type):
    global discriminator
    if "hanoi" in sae.path:
        data = np.loadtxt(sae.local("all_actions.csv"),dtype=np.int8)
    else:
        data = np.loadtxt(sae.local("actions.csv"),dtype=np.int8)
    network, train_in, train_out, test_in, test_out = input_type(data)
    discriminator,_,_ = grid_search(curry(nn_task, network, sae.local("_ad/"),
                                          train_in, train_out, test_in, test_out,),
                                    default_parameters,
                                    parameters)
    discriminator.save()
    
def load():
    global discriminator
    try:
        discriminator = latplan.model.get('PUDiscriminator')(sae.local("_ad/")).load()
    except:
        discriminator = latplan.model.get('Discriminator')(sae.local("_ad/")).load()

def test():
    valid = np.loadtxt(sae.local("all_actions.csv"),dtype=np.int8)
    random.shuffle(valid)
    N = valid.shape[1] // 2
    print("valid",len(valid))

    performance = {}
    
    prediction = np.clip(discriminator.discriminate(valid,batch_size=1000).round(), 0,1)
    performance["type1"] = 100 * np.mean(1-prediction)
    print("type1 error: ",100 * np.mean(1-prediction), "%")

    mixed = generate_oae_action(valid[:1000]) # x2x128 max
    p = latplan.util.puzzle_module(sae.local(""))
    pre_images = sae.decode_binary(mixed[:,:N],batch_size=1000)
    suc_images = sae.decode_binary(mixed[:,N:],batch_size=1000)
    answers = np.array(p.validate_transitions([pre_images, suc_images],batch_size=1000))
    invalid = mixed[np.logical_not(answers)]

    print("mixed",len(mixed), "invalid", len(invalid))

    prediction = np.clip(discriminator.discriminate(invalid,batch_size=1000).round(), 0,1)
    performance["type2"] = 100 * np.mean(prediction)
    print("type2 error: ",100 * np.mean(prediction), "%")

    ind = np.where(np.squeeze(combined(invalid[:,N:])) > 0.5)[0]
    performance["type2/sd"] = 100 * np.mean(prediction[ind])
    print("type2 error (w/o invalid states by sd3): ",100 * np.mean(prediction[ind]), "%")

    ind = p.validate_states(sae.decode_binary(invalid[:,N:],batch_size=1000),verbose=False,batch_size=1000)
    performance["type2/v"] = 100 * np.mean(prediction[ind])
    print("type2 error (w/o invalid states by validator): ",100 * np.mean(prediction[ind]), "%")
    
    import json
    with open(discriminator.local('performance.json'), 'w') as f:
        json.dump(performance, f)

def main(directory, mode="test", input_type="prepare_oae_PU3"):
    from latplan.util import get_ae_type
    global sae, oae, sd3
    sae = latplan.model.get(get_ae_type(directory))(directory).load()
    oae = latplan.model.get('ActionAE')(sae.local("_aae/")).load()
    cae = latplan.model.get('SimpleCAE')(sae.local("_cae/")).load(allow_failure=True)
    sd3 = latplan.model.get('PUDiscriminator')(sae.local("_sd3/")).load()

    if 'learn' in mode:
        learn(eval(input_type))
    else:
        load()

    if 'test' in mode:
        test()

if __name__ == '__main__':
    import sys
    print(sys.argv)
    main(*sys.argv[1:])


################################################################
# unused, unmaintained

def prepare_oae_PU4(known_transisitons):
    print("Learn from pre + action label",
          "*** INCOMPATIBLE MODEL! ***",
          sep="\n")
    N = known_transisitons.shape[1] // 2
    
    y = generate_oae_action(known_transisitons)

    ind = np.where(np.squeeze(combined(y[:,N:])) > 0.5)[0]
    
    y = y[ind]

    actions = oae.encode_action(known_transisitons, batch_size=1000).round()
    positive = np.concatenate((known_transisitons[:,:N], np.squeeze(actions)), axis=1)
    actions = oae.encode_action(y, batch_size=1000).round()
    negative = np.concatenate((y[:,:N], np.squeeze(actions)), axis=1)
    # random.shuffle(negative)
    # negative = negative[:len(positive)]
    # normalize
    return (latplan.model.get('PUDiscriminator'), *prepare_binary_classification_data(positive, negative))

def prepare_oae_PU5(known_transisitons):
    print("Learn from pre + suc + action label",
          "*** INCOMPATIBLE MODEL! ***",
          sep="\n")
    N = known_transisitons.shape[1] // 2
       
    y = generate_oae_action(known_transisitons)
    
    ind = np.where(np.squeeze(combined(y[:,N:])) > 0.5)[0]
    
    y = y[ind]

    actions = oae.encode_action(known_transisitons, batch_size=1000).round()
    positive = np.concatenate((known_transisitons, np.squeeze(actions)), axis=1)
    actions = oae.encode_action(y, batch_size=1000).round()
    negative = np.concatenate((y, np.squeeze(actions)), axis=1)
    # random.shuffle(negative)
    # negative = negative[:len(positive)]
    # normalize
    return (latplan.model.get('PUDiscriminator'), *prepare_binary_classification_data(positive, negative))

def test_oae_pre_label():
    print("--- additional testing on OAE-generated actions")

    known_transisitons = np.loadtxt(sae.local("actions.csv"),dtype=np.int8)
    y = generate_oae_action(known_transisitons)
    N = known_transisitons.shape[1] // 2
    
    answers = np.zeros(len(y),dtype=int)
    p = latplan.util.puzzle_module(sae.local(""))
    batch = 100000
    for i in range(1+len(y)//batch):
        print(i,"/",len(y)//batch)
        pre_images = sae.decode_binary(y[batch*i:batch*(i+1),:N],batch_size=1000)
        suc_images = sae.decode_binary(y[batch*i:batch*(i+1),N:],batch_size=1000)
        answers[batch*i:batch*(i+1)] = np.array(p.validate_transitions([pre_images, suc_images], batch_size=1000)).astype(int)

    # discriminator.report(y, train_data_to=answers) # not appropriate for PUDiscriminator
    actions = oae.encode_action(y, batch_size=1000).round()
    predictions = discriminator.discriminate(np.concatenate((y[:,:N], np.squeeze(actions)), axis=1),batch_size=1000)

    # print("BCE:", bce(predictions, answers))
    # print("type2 error:", 100-mae(predictions.round(), answers)*100, "%")

    ind = np.where(np.squeeze(combined(y[:,N:])) > 0.5)[0]
    print("BCE (w/o invalid states by sd3):", bce(predictions[ind], answers[ind]))
    print("type2 error (w/o invalid states by sd3):", 100-mae(predictions[ind].round(), answers[ind])*100, "%")

    ind = p.validate_states(sae.decode_binary(y[:,N:],batch_size=1000),verbose=False,batch_size=1000)
    print("BCE (w/o invalid states by validator):", bce(predictions[ind], answers[ind]))
    print("type2 error (w/o invalid states by validator):", 100-mae(predictions[ind].round(), answers[ind])*100, "%")

def test_oae_pre_suc_label():
    print("--- additional testing on OAE-generated actions")

    known_transisitons = np.loadtxt(sae.local("actions.csv"),dtype=np.int8)
    y = generate_oae_action(known_transisitons)
    N = known_transisitons.shape[1] // 2
    
    answers = np.zeros(len(y),dtype=int)
    p = latplan.util.puzzle_module(sae.local(""))
    batch = 100000
    for i in range(1+len(y)//batch):
        print(i,"/",len(y)//batch)
        pre_images = sae.decode_binary(y[batch*i:batch*(i+1),:N],batch_size=1000)
        suc_images = sae.decode_binary(y[batch*i:batch*(i+1),N:],batch_size=1000)
        answers[batch*i:batch*(i+1)] = np.array(p.validate_transitions([pre_images, suc_images], batch_size=1000)).astype(int)

    # discriminator.report(y, train_data_to=answers) # not appropriate for PUDiscriminator
    actions = oae.encode_action(y, batch_size=1000).round()
    predictions = discriminator.discriminate(np.concatenate((y, np.squeeze(actions)), axis=1),batch_size=1000)

    # print("BCE:", bce(predictions, answers))
    # print("type2 error:", 100-mae(predictions.round(), answers)*100, "%")

    ind = np.where(np.squeeze(combined(y[:,N:])) > 0.5)[0]
    print("BCE (w/o invalid states by sd3):", bce(predictions[ind], answers[ind]))
    print("type2 error (w/o invalid states by sd3):", 100-mae(predictions[ind].round(), answers[ind])*100, "%")

    ind = p.validate_states(sae.decode_binary(y[:,N:],batch_size=1000),verbose=False,batch_size=1000)
    print("BCE (w/o invalid states by validator):", bce(predictions[ind], answers[ind]))
    print("type2 error (w/o invalid states by validator):", 100-mae(predictions[ind].round(), answers[ind])*100, "%")

def test_artificial():
    valid = np.loadtxt(sae.local("all_actions.csv"),dtype=np.int8)
    random.shuffle(valid)
    valid = valid[:100000]
    
    N = valid.shape[1] // 2
    print("valid",valid.shape)
    discriminator.report(valid,  train_data_to=np.ones((len(valid),)))
    print("type1 error: ",np.mean(1-np.round(discriminator.discriminate(valid,batch_size=1000)))*100, "%")

    c = 0
    def type2(invalid, message):
        nonlocal c
        c += 1
        invalid = set_difference(invalid, valid)
        print("invalid",c,invalid.shape, "---", message)
        discriminator.report(invalid,train_data_to=np.zeros((len(invalid),)))
        print("type2 error:",np.mean(np.round(discriminator.discriminate(invalid,batch_size=1000))) * 100, "%")

        p = latplan.util.puzzle_module(sae.local(""))
        count = 0
        batch = 10000
        for i in range(len(invalid)//batch):
            pre_images = sae.decode_binary(invalid[batch*i:batch*(i+1),:N],batch_size=1000)
            suc_images = sae.decode_binary(invalid[batch*i:batch*(i+1),N:],batch_size=1000)
            validation = p.validate_transitions([pre_images, suc_images])
            count += np.count_nonzero(validation)
        print(count,"valid actions in invalid", c)
    
    type2(np.random.randint(0,2,(len(valid),2*N),dtype=np.int8),
          "invalid actions generated from random bits (both pre and suc)")
        
    type2(generate_random_action(valid, sae),
          "sucessors are random reconstructable states (incl. invalid states such as those with duplicated tiles)")
    
    pre, suc = valid[:,:N], valid[:,N:]
    suc_invalid = np.copy(suc)
    random.shuffle(suc_invalid)
    type2(np.concatenate((pre,suc_invalid),axis=1),
          "generated by swapping successors; pre/suc are correct states from the training examples")

    pre2 = np.loadtxt(sae.local("all_states.csv"),dtype=np.int8)
    suc2 = np.copy(pre2)
    random.shuffle(suc2)
    type2(np.concatenate((pre2,suc2),axis=1),
          "generated by shuffling all valid states; pre/suc are correct but possibly unknown states")
    
    type2(np.concatenate((pre2, pre2), axis=1),
          "invalid actions generated by nop")

    type2(np.concatenate((np.random.randint(0,2,(len(valid),N),dtype=np.int8), suc), axis=1),
          "pre are generated by random bits, suc are correct states from the training examples")

    type2(np.concatenate((pre, np.random.randint(0,2,(len(valid),N),dtype=np.int8)), axis=1),
          "suc are generated by random bits, pre are correct states from the training examples")
    

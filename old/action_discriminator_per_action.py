#!/usr/bin/env python3
import warnings
import config
import numpy as np
import latplan
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

def load_ae(directory):
    global ae
    if ae is None:
        from latplan.util import get_ae_type
        ae = default_networks[get_ae_type(directory)](directory).load()
    return ae

inflation = 1

def repeat_over(array, repeats, axis=0):
    array = np.expand_dims(array, axis)
    array = np.repeat(array, repeats, axis)
    return np.reshape(array,(*array.shape[:axis],-1,*array.shape[axis+2:]))


def prepare_oae_per_action_PU3(known_transisitons):

    print("", sep="\n")
    N = known_transisitons.shape[1] // 2
    states = known_transisitons.reshape(-1, N)

    oae = default_networks['ActionAE'](ae.local("_aae/")).load()
    actions = oae.encode_action(known_transisitons, batch_size=1000).round().astype(int)
    L = actions.shape[2]
    assert L > 1
    
    histogram = np.squeeze(actions.sum(axis=0))
    print(histogram)

    sd3 = default_networks['PUDiscriminator'](ae.local("_sd3/")).load()
    try:
        cae = default_networks['SimpleCAE'](sae.local("_cae/")).load()
        combined_discriminator = default_networks['CombinedDiscriminator'](ae,cae,sd3)
    except:
        combined_discriminator = default_networks['CombinedDiscriminator2'](ae,sd3)

    for label in range(L):
        print("label",label)
        known_transisitons_for_this_label = known_transisitons[np.where(actions[:,:,label] > 0.5)[0]]
        if len(known_transisitons_for_this_label) == 0:
            yield None
        else:        
            _actions = np.zeros((len(states), actions.shape[1], actions.shape[2]), dtype=int)
            _actions[:,:,label] = 1

            y = oae.decode([states,_actions], batch_size=1000).round().astype(np.int8)

            # prune invalid states
            ind = np.where(np.squeeze(combined_discriminator(y[:,N:],batch_size=1000)) > 0.5)[0]
            y = y[ind]

            y = set_difference(y, known_transisitons_for_this_label)
            print(y.shape, known_transisitons_for_this_label.shape)

            train_in, train_out, test_in, test_out = prepare_binary_classification_data(known_transisitons_for_this_label, y)
            yield (train_in, train_out, test_in, test_out)

# default values
default_parameters = {
    'lr'              : 0.0001,
    'batch_size'      : 2000,
    'full_epoch'      : 1000,
    'epoch'           : 1000,
    'max_temperature' : 2.0,
    'min_temperature' : 0.1,
    'M'               : 2,
    'min_grad'        : -0.00001,
}

# exhaustive tuning
parameters = {
    'num_layers' :[1,2],
    'layer'      :[50,300,600],# [400,4000],
    'dropout'    :[0.2,0.5,0.8],    #[0.1,0.4], #0.6,0.7,
    'batch_size' :[4000],
    'full_epoch' :[1000],
    'activation' :['tanh','relu'],
    # quick eval
    'epoch'      :[3000],
    'lr'         :[0.001],
}

# tuned
parameters = {
    'num_layers' :[1],
    'layer'      :[300],# [400,4000],
    'dropout'    :[0.8],    #[0.1,0.4], #0.6,0.7,
    'batch_size' :[4000],
    'full_epoch' :[1000],
    'activation' :['tanh'],
    # quick eval
    'epoch'      :[3000],
    'lr'         :[0.001],
}

def test_oae_generated(directory,discriminator):
    load_ae(directory)
    print("--- additional testing on OAE-generated actions")
    oae = default_networks['ActionAE'](ae.local("_aae/")).load()

    known_transisitons = np.loadtxt(ae.local("actions.csv"),dtype=np.int8)
    actions = oae.encode_action(known_transisitons, batch_size=1000).round()
    histogram = np.squeeze(actions.sum(axis=0,dtype=int))
    print(histogram)
    print(np.count_nonzero(histogram),"actions valid")
    print("valid actions:")
    print(np.where(histogram > 0)[0])
    identified, total = np.squeeze(histogram.sum()), len(actions)
    if total != identified:
        print("network does not explain all actions: only {} out of {} ({}%)".format(
            identified, total, identified * 100 // total ))
    available_actions = np.zeros((np.count_nonzero(histogram), actions.shape[1], actions.shape[2]), dtype=int)

    for i, pos in enumerate(np.where(histogram > 0)[0]):
        available_actions[i][0][pos] = 1

    N = known_transisitons.shape[1] // 2
    states = known_transisitons.reshape(-1, N)
    states = states[:500]
    y = oae.decode([np.repeat(states, len(available_actions), axis=0),
                    np.repeat(available_actions, len(states), axis=0)]) \
           .round().astype(int)

    answers = np.zeros(len(y),dtype=int)
    p = latplan.util.puzzle_module(directory)
    batch = 100000
    for i in range(1+len(y)//batch):
        print(i,"/",len(y)//batch)
        pre_images = ae.decode_binary(y[batch*i:batch*(i+1),:N],batch_size=1000)
        suc_images = ae.decode_binary(y[batch*i:batch*(i+1),N:],batch_size=1000)
        answers[batch*i:batch*(i+1)] = np.array(p.validate_transitions([pre_images, suc_images], batch_size=1000)).astype(int)

    # discriminator.report(y, train_data_to=answers) # not appropriate for PUDiscriminator
    predictions = discriminator.discriminate(y,batch_size=1000)

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

    print("BCE:", bce(predictions, answers))

    sd3 = default_networks['PUDiscriminator'](ae.local("_sd3/")).load()
    from latplan.util import get_ae_type
    from latplan.model import combined_discriminate, combined_discriminate2

    if "conv" not in get_ae_type(directory):
        cae = default_networks['SimpleCAE'](ae.local("_cae/")).load()
        ind = np.where(np.squeeze(combined_discriminate(y[:,N:],ae,cae,sd3,batch_size=1000)) > 0.5)[0]
    else:
        ind = np.where(np.squeeze(combined_discriminate2(y[:,N:],ae,sd3,batch_size=1000)) > 0.5)[0]
    print("BCE (w/o invalid states by sd3):", bce(predictions[ind], answers[ind]))
    print("accuracy (w/o invalid states by sd3):", mae(predictions[ind].round(), answers[ind])*100, "%")

    ind = p.validate_states(ae.decode_binary(y[:,N:],batch_size=1000),verbose=False,batch_size=1000)
    print("BCE (w/o invalid states by validator):", bce(predictions[ind], answers[ind]))
    print("accuracy (w/o invalid states by validator):", mae(predictions[ind].round(), answers[ind])*100, "%")
    

def main(directory, mode, input_type=prepare_oae_per_action_PU3):
    directory_ad = "{}/_ads/".format(directory)
    print(directory, mode, input_type)

    try:
        if 'learn' in mode:
            raise Exception('learn')
        if input_type is prepare_oae_validated:
            discriminator = default_networks['Discriminator'](directory_ad).load()
        else:
            discriminator = default_networks['PUDiscriminator'](directory_ad).load()
    except:
        data = np.loadtxt("{}/actions.csv".format(directory),dtype=np.int8)
        load_ae(directory)
        discriminators = []
        evaluations    = []
        for i, train_test_data in enumerate(input_type(data)):
            print("label",i)
            try:
                d,_,e = grid_search(curry(nn_task, default_networks['PUDiscriminator'],
                                          directory_ad+str(i)+"/",
                                          *train_test_data),
                                    default_parameters,
                                    parameters)
                discriminators.append(d)
                evaluations.append(e)
            except Exception as e:
                print(e)
                discriminators.append(None)
                evaluations.append(None)
            print(evaluations)
        
    # test_oae_generated(directory,discriminators)


if __name__ == '__main__':
    import sys
    def myeval(str):
        try:
            return eval(str)
        except:
            return str
    print(list(map(myeval,sys.argv[1:])))
    main(*map(myeval,sys.argv[1:]))

#!/usr/bin/env python3
import warnings
import config
import sys
import numpy as np
import numpy.random as random
import latplan
import latplan.model
from latplan.model       import combined_sd
from latplan.util        import curry, prepare_binary_classification_data, set_difference, union
from latplan.util.tuning import grid_search, nn_task
from latplan.util.np_distances import *

from keras.optimizers import Adam
from keras_adabound   import AdaBound
from keras_radam      import RAdam

import keras.optimizers

setattr(keras.optimizers,"radam", RAdam)
setattr(keras.optimizers,"adabound", AdaBound)

import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(threshold=sys.maxsize,formatter={'float_kind':float_formatter})

def generate_random(data,sae,batch=None):
    import sys
    threshold = sys.float_info.epsilon
    rate_threshold = 0.99
    max_repeat = 50

    def regenerate(sae, data):
        images           = sae.decode(data,batch_size=2000)
        data_invalid_rec = sae.encode(images,batch_size=2000)
        return data_invalid_rec

    def regenerate_many(sae, data):
        loss = 1000000000
        for i in range(max_repeat):
            data_rec = regenerate(sae, data)
            prev_loss = loss
            loss    = bce(data,data_rec)
            if len(data) > 3000:
                print(loss, loss / prev_loss)
            data = data_rec
            if (loss / prev_loss > rate_threshold):
                if len(data) > 3000:
                    print("improvement saturated: loss / prev_loss = ", loss / prev_loss, ">", rate_threshold)
                break
            if loss <= threshold:
                print("good amount of loss:", loss, "<", threshold)
                break
        return data.round().astype(np.int8)
    
    def prune_unreconstructable(sae,data):
        rec = regenerate(sae,data)
        loss = bce(data,rec,(1,))
        return data[np.where(loss < threshold)[0]]

    if batch is None:
        batch = data.shape[0]
    N     = data.shape[1]
    data_invalid = random.randint(0,2,(batch,N),dtype=np.int8)
    data_invalid = regenerate_many(sae, data_invalid)
    data_invalid = prune_unreconstructable(sae, data_invalid)
    data_invalid = set_difference(data_invalid.round(), data.round())
    return data_invalid

def prepare(data_valid, sae):
    gen_batch = 10000 if len(data_valid) < 2000 else None
    data_mixed = generate_random(data_valid, sae, gen_batch)
    try:
        p = 0
        pp = 0
        ppp = 0
        i = 0
        limit = 600
        while len(data_mixed) < len(data_valid) and p < len(data_mixed) and i < limit:
            i += 1
            p = pp
            pp = ppp
            ppp = len(data_mixed)
            data_mixed = union(data_mixed, generate_random(data_valid, sae, gen_batch))
            print("valid:",len(data_valid),
                  "mixed:", len(data_mixed),
                  "iteration:", i, "/", limit,
                  "## generation stops when it failed to generate new examples three times in a row")
    except KeyboardInterrupt:
        pass
    finally:
        print("generation stopped")

    if len(data_valid) < len(data_mixed):
        # downsample
        data_mixed = data_mixed[:len(data_valid)]
    elif len(data_mixed) == 0:
        data_mixed = data_valid.copy() # valid data are also mixed data by definition
    else:
        # oversample
        data_mixed = np.repeat(data_mixed, 1+(len(data_valid)//len(data_mixed)), axis=0)
        data_mixed = data_mixed[:len(data_valid)]

    train_in, train_out, test_in, test_out = prepare_binary_classification_data(data_valid, data_mixed)
    return train_in, train_out, test_in, test_out, data_valid, data_mixed

def prepare_random(data_valid, sae, inflation=1):
    batch = data_valid.shape[0]
    data_mixed = random.randint(0,2,data_valid.shape,dtype=np.int8)
    train_in, train_out, test_in, test_out = prepare_binary_classification_data(data_valid, data_mixed)
    return train_in, train_out, test_in, test_out, data_valid, data_mixed

sae = None
cae = None
discriminator = None

def learn(method):
    global cae, discriminator
    default_parameters = {
        'lr'              : 0.001,
        'batch_size'      : 1000,
        'epoch'           : 1000,
        'max_temperature' : 2.0,
        'min_temperature' : 0.1,
        'M'               : 2,
        'min_grad'        : 0.0,
        'optimizer'       : 'radam',
        'dropout'         : 0.4,
    }
    data_valid = np.loadtxt(sae.local("states.csv"),dtype=np.int8)
    train_in, train_out, test_in, test_out, data_valid, data_mixed = prepare(data_valid,sae)
    sae.plot_autodecode(data_mixed[:8], sae.local("_sd3/fake_samples.png"))

    def save(net):
        net.parameters["method"] = method
        net.save()

    if method == "cae":
        # decode into image, learn a separate cae and learn from it
        train_image, test_image = sae.decode(train_in), sae.decode(test_in)
        cae,_,_ = grid_search(curry(nn_task, latplan.model.get('SimpleCAE'),
                                    sae.local("_cae"),
                                    train_image, train_image, test_image, test_image),
                              default_parameters,
                              {
                                  'num_layers' :[1,2],
                                  'layer'      :[300,1000],
                                  'clayer'     :[16],
                                  'activation' :['relu','tanh'],
                              },
                              sae.local("_cae/"),
                              report_best= save, shuffle=False,
        )
        cae.save()
        train_in2, test_in2 = cae.encode(train_image), cae.encode(test_image)
        discriminator,_,_ = grid_search(curry(nn_task, latplan.model.get('PUDiscriminator'), sae.local("_sd3/"),
                                              train_in2, train_out, test_in2, test_out,),
                                        default_parameters,
                                        {
                                            'num_layers' :[1,2],
                                            'layer'      :[300,1000],
                                            'clayer'     :[16],
                                            'activation' :['relu','tanh'],
                                        },
                                        sae.local("_sd3/"),
                                        report_best= save, shuffle=False,
        )
    if method == "direct":
        # learn directly from the latent encoding
        discriminator,_,_ = grid_search(curry(nn_task, latplan.model.get('PUDiscriminator'), sae.local("_sd3/"),
                                              train_in, train_out, test_in, test_out,),
                                        default_parameters,
                                        {
                                            'num_layers' :[1,2],
                                            'layer'      :[300,1000],# [400,4000],
                                            'activation' :['relu','tanh'],
                                        },
                                        sae.local("_sd3/"),
                                        report_best= save, shuffle=False,
        )
    if method == "image":
        # learn directly from the image
        train_image, test_image = sae.decode(train_in), sae.decode(test_in)
        discriminator,_,_ = grid_search(curry(nn_task, latplan.model.get('PUDiscriminator'), sae.local("_sd3/"),
                                              train_image, train_out, test_image, test_out,),
                                        default_parameters,
                                        {
                                            'num_layers' :[1,2],
                                            'layer'      :[300,1000],# [400,4000],
                                            'activation' :['relu','tanh'],
                                        },
                                        sae.local("_sd3/"),
                                        report_best= save, shuffle=False,
        )

def load(method):
    global cae, discriminator
    if method == "feature":
        discriminator = latplan.model.get('PUDiscriminator')(sae.local("_sd3/")).load()
    if method == "cae":
        cae = latplan.model.get('SimpleCAE'    )(sae.local("_cae")).load()
        discriminator = latplan.model.get('PUDiscriminator')(sae.local("_sd3/")).load()
    if method == "direct":
        discriminator = latplan.model.get('PUDiscriminator')(sae.local("_sd3/")).load()
    if method == "image":
        discriminator = latplan.model.get('PUDiscriminator')(sae.local("_sd3/")).load()
    assert method == discriminator.parameters["method"]

def test(method):
    valid = np.loadtxt(sae.local("valid_actions.csv"),dtype=np.int8)
    N = valid.shape[1]//2
    valid = valid.reshape([-1,N])
    random.shuffle(valid)
    invalid = np.loadtxt(sae.local("invalid_actions.csv"),dtype=np.int8)
    invalid = invalid.reshape([-1, N])
    random.shuffle(invalid)
    mixed = np.loadtxt(sae.local("fake_actions.csv"),dtype=np.int8)
    mixed = mixed.reshape([-1, N])
    random.shuffle(mixed)

    performance = {}
    def reg(names,value,d=performance):
        name = names[0]
        if len(names)>1:
            try:
                tmp = d[name]
            except KeyError:
                tmp={}
                d[name]=tmp
            reg(names[1:], value, tmp)
        else:
            d[name] = float(value)
            print(name,": ", value)
    
    reg(["valid"],   len(valid))
    reg(["mixed"],   len(mixed))
    reg(["invalid"], len(invalid))

    def measure(valid, invalid, suffix):
        minlen=min(len(valid),len(invalid))
        
        valid_tmp   = valid  [:minlen]
        invalid_tmp = invalid[:minlen]
        
        tp = np.clip(combined_sd(valid_tmp  ,sae,cae,discriminator,batch_size=1000).round(), 0,1) # true positive
        fp = np.clip(combined_sd(invalid_tmp,sae,cae,discriminator,batch_size=1000).round(), 0,1) # false positive
        tn = 1-fp
        fn = 1-tp
    
        reg([suffix,"minlen"     ],minlen)
        recall      = np.mean(tp) # recall / sensitivity / power / true positive rate out of condition positive
        specificity = np.mean(tn) # specificity / true negative rate out of condition negative
        reg([suffix,"recall"     ],recall)
        reg([suffix,"specificity"],specificity)
        reg([suffix,"f"],(2*recall*specificity)/(recall+specificity))
        try:
            reg([suffix,"precision"  ],np.sum(tp)/(np.sum(tp)+np.sum(fp)))
        except ZeroDivisionError:
            reg([suffix,"precision"  ],float('nan'))
        try:
            reg([suffix,"accuracy"   ],(np.sum(tp)+np.sum(tn))/(2*minlen))
        except ZeroDivisionError:
            reg([suffix,"accuracy"   ],float('nan'))
        return
    
    measure(valid,invalid,"raw")
    
    import json
    with open(discriminator.local('performance.json'), 'w') as f:
        json.dump(performance, f)

def main(directory, mode="test", method='direct'):
    global sae
    sae = latplan.model.load(directory)
    sae.single_mode()
    import subprocess
    subprocess.call(["mkdir","-p", sae.local("_sd3/")])

    if 'learn' in mode:
        learn(method)
    
    load(method)

    if 'test' in mode:
        test(method)

if __name__ == '__main__':
    import sys
    try:
        print(sys.argv)
        main(*sys.argv[1:])
    except:
        import latplan.util.stacktrace
        latplan.util.stacktrace.format()
        sys.exit("{} [directory] [mode=test] [method=feature]".format(sys.argv[0]))

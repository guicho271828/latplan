#!/usr/bin/env python3
import warnings
import config
import numpy as np
import latplan
import latplan.model
from latplan.model import combined_discriminate, combined_discriminate2
from latplan.util        import curry, prepare_binary_classification_data, set_difference, union, bce
from latplan.util.tuning import grid_search, nn_task

import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def generate_random(data,sae,batch=None):
    import sys
    threshold = sys.float_info.epsilon
    rate_threshold = 0.99
    max_repeat = 50

    def regenerate(sae, data):
        images           = sae.decode_binary(data,batch_size=2000)
        data_invalid_rec = sae.encode_binary(images,batch_size=2000)
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
    data_invalid = np.random.randint(0,2,(batch,N),dtype=np.int8)
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
    data_mixed = np.random.randint(0,2,data_valid.shape,dtype=np.int8)
    train_in, train_out, test_in, test_out = prepare_binary_classification_data(data_valid, data_mixed)
    return train_in, train_out, test_in, test_out, data_valid, data_mixed

sae = None
cae = None
discriminator = None
combined = None

def learn(method):
    global cae, discriminator
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
    data_valid = np.loadtxt(sae.local("states.csv"),dtype=np.int8)
    train_in, train_out, test_in, test_out, data_valid, data_mixed = prepare(data_valid,sae)
    sae.plot_autodecode(data_mixed[:8], sae.local("_sd3/fake_samples.png"))

    if method == "feature":
        # decode into image, extract features and learn from it
        train_image, test_image = sae.decode_binary(train_in), sae.decode_binary(test_in)
        train_in2, test_in2 = sae.get_features(train_image), sae.get_features(test_image)
        discriminator,_,_ = grid_search(curry(nn_task, latplan.model.get('PUDiscriminator'), sae.local("_sd3/"),
                                              train_in2, train_out, test_in2, test_out,),
                                        default_parameters,
                                        {
                                            'num_layers' :[1],
                                            'layer'      :[50],
                                            'clayer'     :[16],
                                            'dropout'    :[0.8],
                                            'batch_size' :[1000],
                                            'full_epoch' :[1000],
                                            'activation' :['relu'],
                                            'epoch'      :[3000],
                                            'lr'         :[0.0001],
                                        })
    if method == "cae":
        # decode into image, learn a separate cae and learn from it
        train_image, test_image = sae.decode_binary(train_in), sae.decode_binary(test_in)
        cae,_,_ = grid_search(curry(nn_task, latplan.model.get('SimpleCAE'),
                                    sae.local("_cae"),
                                    train_image, train_image, test_image, test_image),
                              default_parameters,
                              {
                                  'num_layers' :[2],
                                  'layer'      :[500],
                                  'clayer'     :[16],
                                  'dropout'    :[0.4],
                                  'batch_size' :[4000],
                                  'full_epoch' :[1000],
                                  'activation' :['relu'],
                                  'epoch'      :[30],
                                  'lr'         :[0.001],
                              })
        cae.save()
        train_in2, test_in2 = cae.encode(train_image), cae.encode(test_image)
        discriminator,_,_ = grid_search(curry(nn_task, latplan.model.get('PUDiscriminator'), sae.local("_sd3/"),
                                              train_in2, train_out, test_in2, test_out,),
                                        default_parameters,
                                        {
                                            'num_layers' :[1],
                                            'layer'      :[50],
                                            'clayer'     :[16],
                                            'dropout'    :[0.8],
                                            'batch_size' :[1000],
                                            'full_epoch' :[1000],
                                            'activation' :['relu'],
                                            'epoch'      :[3000],
                                            'lr'         :[0.0001],
                                        })
    if method == "direct":
        # learn directly from the latent encoding
        discriminator,_,_ = grid_search(curry(nn_task, latplan.model.get('PUDiscriminator'), sae.local("_sd3/"),
                                              train_in, train_out, test_in, test_out,),
                                        default_parameters,
                                        {
                                            'layer'      :[300],# [400,4000],
                                            'dropout'    :[0.1], #[0.1,0.4],
                                            'num_layers' :[2],
                                            'batch_size' :[1000],
                                            'full_epoch' :[1000],
                                            'activation' :['tanh'],
                                            # quick eval
                                            'epoch'      :[200],
                                            'lr'         :[0.0001],
                                        })
    if method == "image":
        # learn directly from the image
        train_image, test_image = sae.decode_binary(train_in), sae.decode_binary(test_in)
        discriminator,_,_ = grid_search(curry(nn_task, latplan.model.get('PUDiscriminator'), sae.local("_sd3/"),
                                              train_image, train_out, test_image, test_out,),
                                        default_parameters,
                                        {
                                            'layer'      :[300],# [400,4000],
                                            'dropout'    :[0.1], #[0.1,0.4],
                                            'num_layers' :[2],
                                            'batch_size' :[1000],
                                            'full_epoch' :[1000],
                                            'activation' :['tanh'],
                                            # quick eval
                                            'epoch'      :[200],
                                            'lr'         :[0.0001],
                                        })
    discriminator.parameters["method"] = method
    discriminator.save()

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

def load2(method):
    global combined
    if method == "feature":
        def d (states):
            return combined_discriminate2(states,sae,discriminator,batch_size=1000).round()
        combined = d
    if method == "cae":
        def d (states):
            return combined_discriminate(states,sae,cae,discriminator,batch_size=1000).round()
        combined = d
    if method == "direct":
        def d (states):
            return discriminator.discriminate(states,batch_size=1000).round()
        combined = d
    if method == "image":
        def d (states):
            images = sae.decode_binary(data,batch_size=1000)
            return discriminator.discriminate(images,batch_size=1000).round()
        combined = d

def test(method):
    states_valid = np.loadtxt(sae.local("all_states.csv"),dtype=np.int8)
    print("valid",states_valid.shape)

    from latplan.util.plot import plot_grid
    performance = {}

    ################################################################
    # type 1 error
    type1_d = combined(states_valid)
    type1_error = np.sum(1- type1_d)
    performance["type1"] = type1_error/len(states_valid) * 100
    print("type1 error:",type1_error,"/",len(states_valid),
          "Error ratio:", type1_error/len(states_valid) * 100, "%")

    type1_error_images = sae.decode_binary(states_valid[np.where(type1_d < 0.1)[0]])[:120]
    if len(type1_error_images) == 0:
        print("We observed ZERO type1-error! Hooray!")
    else:
        plot_grid(type1_error_images,
                  w=20,
                  path=discriminator.local("type1_error.png"))

    ################################################################
    # type 2 error
    _,_,_,_, _, states_mixed = prepare(states_valid[:50000],sae)
    print(len(states_mixed),"reconstructable states generated.")
    performance["reconstructable_states"] = len(states_mixed)

    p = latplan.util.puzzle_module(sae.path)
    is_valid = p.validate_states(sae.decode_binary(states_mixed))
    states_invalid = states_mixed[np.logical_not(is_valid)]
    states_invalid = states_invalid[:30000]
    print(len(states_invalid),"invalid states generated.")
    performance["invalid_states"] = len(states_invalid)

    if len(states_invalid) == 0:
        performance["type2"] = float('nan')
        print("We observed ZERO invalid states.")
    else:
        plot_grid(sae.decode_binary(states_invalid)[:120],
              w=20,
              path=discriminator.local("surely_invalid_states.png"))
        type2_d = combined(states_invalid)
        type2_error = np.sum(type2_d)
        performance["type2"] = type2_error/len(states_invalid) * 100
        print("type2 error:",type2_error,"/",len(states_invalid),
              "Error ratio:", type2_error/len(states_invalid) * 100, "%")

        type2_error_images = sae.decode_binary(states_invalid[np.where(type2_d > 0.9)[0]])[:120]
        if len(type2_error_images) == 0:
            print("We observed ZERO type2-error! Hooray!")
        else:
            plot_grid(type2_error_images,
                      w=20,
                      path=discriminator.local("type2_error.png"))
    
    import json
    with open(discriminator.local('performance.json'), 'w') as f:
        json.dump(performance, f)

def main(directory, mode="test", method='feature'):
    from latplan.util import get_ae_type
    global sae
    sae = latplan.model.get(get_ae_type(directory))(directory).load()
    import subprocess
    subprocess.call(["mkdir","-p", sae.local("_sd3/")])

    if 'learn' in mode:
        learn(method)
    else:
        load(method)

    load2(method)

    if 'test' in mode:
        test(method)

if __name__ == '__main__':
    import sys
    try:
        print(sys.argv)
        main(*sys.argv[1:])
    except:
        import traceback
        print(traceback.format_exc())
        sys.exit("{} [directory] [mode=test] [method=feature]".format(sys.argv[0]))

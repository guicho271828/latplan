import numpy as np
import os.path
from ..util.tuning import parameters
from .common import *
from .normalization import normalize_transitions
from . import common


@register
def lightsout(type='digital',size=4,num_examples=6500,N=None,num_actions=None,aeclass="ConvolutionalGumbelAE",comment=""):
    for name, value in locals().items():
        if value is not None:
            parameters[name] = [value]
    parameters["aeclass"] = aeclass

    import importlib
    generator = 'latplan.puzzles.lightsout_{}'.format(type)
    parameters["generator"] = generator
    p = importlib.import_module(generator)
    p.setup()
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["lightsout",type,size]))+".npz")
    with np.load(path) as data:
        pre_configs = data['pres'][:num_examples]
        suc_configs = data['sucs'][:num_examples]
    pres = p.generate(pre_configs)
    sucs = p.generate(suc_configs)
    pres = pres.reshape([*pres.shape,1])
    sucs = sucs.reshape([*sucs.shape,1])
    assert len(pres.shape) == 4

    transitions, states = normalize_transitions(pres, sucs)

    ae = run(os.path.join("samples",common.sae_path), transitions)


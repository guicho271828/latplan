import numpy as np
import os.path
from ..util.tuning import parameters
from .common import *
from .normalization import normalize_transitions
from . import common


@register
def hanoi(disks=7,towers=4,num_examples=6500,N=None,num_actions=None,aeclass="ConvolutionalGumbelAE",comment=""):
    for name, value in locals().items():
        if value is not None:
            parameters[name] = [value]
    parameters["aeclass"] = aeclass
    parameters["generator"] = "latplan.puzzles.hanoi"

    import latplan.puzzles.hanoi as p
    p.setup()
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["hanoi",disks,towers]))+".npz")
    with np.load(path) as data:
        pre_configs = data['pres'][:num_examples]
        suc_configs = data['sucs'][:num_examples]
    pres = p.generate(pre_configs,disks,towers)
    sucs = p.generate(suc_configs,disks,towers)
    assert len(pres.shape) == 4

    transitions, states = normalize_transitions(pres, sucs)

    ae = run(os.path.join("samples",common.sae_path), transitions)




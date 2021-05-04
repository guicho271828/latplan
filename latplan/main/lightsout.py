import numpy as np
import os.path
from ..util.tuning import parameters
from .common import *
from .normalization import normalize_transitions
from . import common

def lightsout(args):
    type = args.type
    size = args.size
    num_examples = args.num

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


_parser = subparsers.add_parser('lightsout', formatter_class=argparse.ArgumentDefaultsHelpFormatter, help='LightsOut game (see https://en.wikipedia.org/wiki/Lights_Out_(game))')
_parser.add_argument('type', choices=["digital","twisted"], help='When twisted, the screen is corrupted by the swirl effect.')
_parser.add_argument('size', type=int, default=4, help='The size of the grid.')
add_common_arguments(_parser,lightsout)


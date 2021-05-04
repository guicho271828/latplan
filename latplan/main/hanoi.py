import numpy as np
import os.path
from ..util.tuning import parameters
from .common import *
from .normalization import normalize_transitions
from . import common


def hanoi(args):
    parameters["generator"] = "latplan.puzzles.hanoi"
    disks  = args.disks
    towers = args.towers
    num_examples = args.num_examples

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



_parser = subparsers.add_parser('hanoi', formatter_class=argparse.ArgumentDefaultsHelpFormatter, help='Tower of Hanoi.')
_parser.add_argument('disks', type=int, default=7, help='The number of disks in the environment.')
_parser.add_argument('towers', type=int, default=4, help='The number of towers, or the width of the environment.')
add_common_arguments(_parser,hanoi)

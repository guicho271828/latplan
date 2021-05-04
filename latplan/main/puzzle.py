import numpy as np
import os.path
from ..util.tuning import parameters
from .common import *
from .normalization import normalize_transitions, normalize_transitions_objects
from . import common
from ..puzzles.objutil import bboxes_to_coord, random_object_masking, tiled_bboxes, image_to_tiled_objects
from ..util.stacktrace import format

def load_puzzle(type,width,height,num_examples,objects=True,**kwargs):
    import importlib
    generator = 'latplan.puzzles.puzzle_{}'.format(type)
    parameters["generator"] = generator
    p = importlib.import_module(generator)
    p.setup()
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["puzzle",type,width,height]))+".npz")
    with np.load(path) as data:
        pres_configs = data['pres'][:num_examples]
        sucs_configs = data['sucs'][:num_examples]

    pres = p.states(width, height, pres_configs)[:,:,:,None]
    sucs = p.states(width, height, sucs_configs)[:,:,:,None]
    B, H, W, C = pres.shape
    parameters["picsize"]        = [[H,W]]
    print("loaded. picsize:",[H,W])

    if objects:
        pres = image_to_tiled_objects(pres, p.setting['base'])
        sucs = image_to_tiled_objects(sucs, p.setting['base'])
        bboxes = tiled_bboxes(B, height, width, p.setting['base'])
        pres = np.concatenate([pres,bboxes], axis=-1)
        sucs = np.concatenate([sucs,bboxes], axis=-1)
        transitions, states = normalize_transitions_objects(pres,sucs,**kwargs)
    else:
        transitions, states = normalize_transitions(pres, sucs)
    return transitions, states


################################################################
# flat images

def puzzle(args):
    transitions, states = load_puzzle(**vars(args),objects=False)

    ae = run(os.path.join("samples",common.sae_path), transitions)


_parser = subparsers.add_parser('puzzle', formatter_class=argparse.ArgumentDefaultsHelpFormatter, help='Sliding tile puzzle.')
_parser.add_argument('type', choices=["mnist","mandrill","spider"], help='')
_parser.add_argument('width', type=int, default=3, help='Integer width of the puzzle. In MNIST, the value must be 3.')
_parser.add_argument('height', type=int, default=3, help='Integer height of the puzzle. In MNIST, the value must be 3.')
add_common_arguments(_parser,puzzle)

################################################################
# object-based representation

def puzzle_objs(args):
    transitions, states = load_puzzle(**vars(args))

    ae = run(os.path.join("samples",common.sae_path), transitions)

    transitions = transitions[:6]
    _,_,O,_ = transitions.shape
    print("plotting interpolation")
    for O2 in [ 3, 5, 7 ]:
        try:
            masked2 = random_object_masking(transitions,O2)
        except Exception as e:
            print(f"O2={O2}. Masking failed due to {e}, skip this iteration.")
            continue
        ae.reload_with_shape(masked2.shape[1:])
        plot_autoencoding_image(ae,masked2,f"interpolation-{O2}")

    pass


_parser = subparsers.add_parser('puzzle_objs', formatter_class=argparse.ArgumentDefaultsHelpFormatter, help='Object-based sliding tile puzzle.')
_parser.add_argument('type', choices=["mnist","mandrill","spider"], help='')
_parser.add_argument('width', type=int, default=3, help='Integer width of the puzzle. In MNIST, the value must be 3.')
_parser.add_argument('height', type=int, default=3, help='Integer height of the puzzle. In MNIST, the value must be 3.')
add_common_arguments(_parser,puzzle_objs,True)


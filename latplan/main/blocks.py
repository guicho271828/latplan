import numpy as np
import os.path
from ..util.tuning import parameters
from .common import *
from .normalization import normalize_transitions, normalize_transitions_objects
from ..puzzles.objutil import bboxes_to_coord, random_object_masking
from . import common
from ..util.stacktrace import format

def load_blocks(track,num_examples,objects=True,**kwargs):
    with np.load(os.path.join(latplan.__path__[0],"puzzles",track+".npz")) as data:
        images               = data['images'].astype(np.float32) / 255
        bboxes               = data['bboxes']
        all_transitions_idx  = data['transitions']
        picsize              = data['picsize']
        num_states, num_objs = bboxes.shape[0:2]
        print("loaded. picsize:",picsize)

    parameters["picsize"] = [picsize.tolist()]
    parameters["generator"] = None

    all_transitions_idx = all_transitions_idx.reshape((len(all_transitions_idx)//2, 2))
    np.random.shuffle(all_transitions_idx)
    transitions_idx = all_transitions_idx[:num_examples]

    if objects:
        all_states = np.concatenate((images.reshape((num_states, num_objs, -1)),
                                     bboxes.reshape((num_states, num_objs, -1))),
                                    axis = -1)
        pres = all_states[transitions_idx[:,0]]
        sucs = all_states[transitions_idx[:,1]]
        transitions, states = normalize_transitions_objects(pres,sucs,**kwargs)
    else:
        pres = images[transitions_idx[:,0],0]
        sucs = images[transitions_idx[:,1],0]
        transitions, states = normalize_transitions(pres,sucs)

    return transitions, states


################################################################
# flat images

def blocks(args):
    parameters["generator"] = "latplan.puzzles.blocks"
    transitions, states = load_blocks(**vars(args),objects=False)

    ae = run(os.path.join("samples",common.sae_path), transitions)


_parser = subparsers.add_parser('blocks',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help='Blocksworld environment. Requires an archive made by github.com/IBM/photorealistic-blocksworld')
_parser.add_argument('track', help='Name of the archive stored in latplan/puzzles/. Example: blocks-3-flat')
add_common_arguments(_parser,blocks)

################################################################
# object-based representation

def blocks_objs(args):
    parameters["generator"] = "latplan.puzzles.blocks"
    transitions, states = load_blocks(**vars(args))

    ae = run(os.path.join("samples",common.sae_path), transitions)

    transitions = transitions[:6]
    _,_,O,_ = transitions.shape
    print("plotting interpolation")
    for O2 in [3,4,5]:
        try:
            masked2 = random_object_masking(transitions,O2)
        except Exception as e:
            print(f"O2={O2}. Masking failed due to {e}, skip this iteration.")
            continue
        ae.reload_with_shape(masked2.shape[1:])
        plot_autoencoding_image(ae,masked2,f"interpolation-{O2}")
    print("plotting extrapolation")
    # for i in [9.12,15]:
    #     if f"blocks-{i}-objs" == track:
    #         continue
    #     transitions, states = load_blocks(f"blocks-{i}-objs",num_examples)
    #     transitions = transitions[:6]
    #     ae.parameters["picsize"] = parameters["picsize"][0]
    #     ae.reload_with_shape(transitions.shape[1:])
    #     plot_autoencoding_image(ae,transitions,f"extrapolation-{i}")
    pass


_parser = subparsers.add_parser('blocks_objs',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help='Object-based blocksworld environment. Requires an archive made by github.com/IBM/photorealistic-blocksworld')
_parser.add_argument('track', help='Name of the archive stored in latplan/puzzles/. blocks-3-objs')
add_common_arguments(_parser,blocks_objs,True)

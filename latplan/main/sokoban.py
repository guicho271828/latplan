import numpy as np
import os.path
from ..util.tuning import parameters
from .common import *
from .normalization import  normalize_transitions, normalize_transitions_objects
from ..puzzles.objutil import bboxes_to_coord, random_object_masking
from . import common
from ..util.stacktrace import format

def load_sokoban(track,num_examples,objects=True,**kwargs):
    with np.load(os.path.join(latplan.__path__[0],"puzzles",track+".npz")) as data:
        pres   = data['pres'][:num_examples] / 255 # [B,O,sH*sW*C]
        sucs   = data['sucs'][:num_examples] / 255 # [B,O,sH*sW*C]
        bboxes = data['bboxes'][:num_examples] # [B,O,4]
        picsize = data['picsize']
        print("loaded. picsize:",picsize)

    parameters["picsize"] = [picsize.tolist()]
    parameters["generator"] = None

    if objects:
        pres = np.concatenate([pres,bboxes], axis=-1)
        sucs = np.concatenate([sucs,bboxes], axis=-1)
        transitions, states = normalize_transitions_objects(pres,sucs,**kwargs)
    else:
        pres = pres[:,0].reshape((-1, *picsize))
        sucs = sucs[:,0].reshape((-1, *picsize))
        transitions, states = normalize_transitions(pres,sucs)
    return transitions, states


################################################################
# flat images

def sokoban(args):
    parameters["generator"] = "latplan.puzzles.sokoban"
    transitions, states = load_sokoban(**vars(args),objects=False)

    ae = run(os.path.join("samples",common.sae_path), transitions)


_parser = subparsers.add_parser('sokoban', formatter_class=argparse.ArgumentDefaultsHelpFormatter, help='Sokoban environment rendered by PDDLGym.')
_parser.add_argument('track', help='Name of the archive stored in latplan/puzzles/. Example: sokoban_image-10000-global-global-0-train')
add_common_arguments(_parser,sokoban)

################################################################
# object-based representation

def sokoban_objs(args):
    parameters["generator"] = "latplan.puzzles.sokoban"
    transitions, states = load_sokoban(**vars(args))

    ae = run(os.path.join("samples",common.sae_path), transitions)

    transitions = transitions[:6]
    _,_,O,_ = transitions.shape
    print("plotting interpolation")
    for O2 in set([int(O*i/5.0) for i in range(1,5)]):
        try:
            masked2 = random_object_masking(transitions,O2)
        except Exception as e:
            print(f"O2={O2}. Masking failed due to {e}, skip this iteration.")
            continue
        ae.reload_with_shape(masked2.shape[1:])
        plot_autoencoding_image(ae,masked2,f"interpolation-{O2}")
    print("plotting extrapolation")
    for i in range(4):
        if str(i) == track.split("-")[-2]:
            continue
        transitions, states = load_sokoban(f"sokoban_image-1000-global-object-{i}-test",num_examples)
        transitions = transitions[:6] # only a small amount of examples are used for plotting
        ae.parameters["picsize"] = parameters["picsize"][0]
        ae.reload_with_shape(transitions.shape[1:])
        plot_autoencoding_image(ae,transitions,f"extrapolation-{i}")
    pass


_parser = subparsers.add_parser('sokoban_objs', formatter_class=argparse.ArgumentDefaultsHelpFormatter, help='Object-based Sokoban environment rendered by PDDLGym.')
_parser.add_argument('track', help='Name of the archive stored in latplan/puzzles/. Example: sokoban_image-10000-global-object-merged-train')
add_common_arguments(_parser,sokoban_objs,True)


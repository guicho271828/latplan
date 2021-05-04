import numpy as np
import numpy.random as random
from ..util.tuning import parameters
from ..puzzles.objutil import *


def normalize(x,save=True):
    if ("mean" in parameters) and save:
        mean = np.array(parameters["mean"][0])
        std  = np.array(parameters["std"][0])
    else:
        mean               = np.mean(x,axis=0)
        std                = np.std(x,axis=0)
        if save:
            parameters["mean"] = [mean.tolist()]
            parameters["std"]  = [std.tolist()]
    print("normalized shape:",mean.shape,std.shape)
    return (x - mean)/(std+1e-20)


def unnormalize(x):
    mean                       = np.array(parameters["mean"][0])
    std                        = np.array(parameters["std"][0])
    x                          = (x * std)+mean
    return x


def normalize_transitions(pres,sucs):
    """Normalize a dataset for image-based input format.
Normalization is performed across batches."""
    B, *F = pres.shape
    transitions = np.stack([pres,sucs], axis=1) # [B, 2, F]
    print(transitions.shape)

    normalized  = normalize(np.reshape(transitions, [-1, *F])) # [2BO, F]
    states      = normalized.reshape([-1, *F])
    transitions = states.reshape([-1, 2, *F])
    return transitions, states


def normalize_transitions_objects(pres,sucs,object_ordering=None,randomize_locations=False,location_representation="coord",**kwargs):
    """
Normalize a dataset for object-based transition input format [B, 2, O, F] where B : batch , O : object, F: feature.
Normalization is performed across B,O dimensions, e.g., mean/variance has shape [F].

object_ordering specifies the ordering of objects.

None : maintain the original order in the archive.
"random"  : objects are shuffled individually in each state.
            predecessor and successor state pair may have a different object ordering.
"match_images_sinkhorn": compute pairwise L2 distances between images of O objects in the predecessor and successor states.
            we then compute optimal transport using sinkhorn iteration
"match_images_argmin": compute pairwise L2 distances between images of O objects in the predecessor and successor states.
            we then compute argmin
"match_bbox_argmin": compute pairwise L2 distances between bboxes of O objects in the predecessor and successor states.
            we then compute argmin

location_representation specifies the representation for network input/output, which one of the following:

"coord" : Represent the location as a center location (cx,cy) and a dimension (w,h).

"bbox"  : Represent the location as a bounding box (x1,y1,x2,y2).

"binary": Represent the location in a k-bit binary encoding by dividing each coordinate into 2^k cells.

"sinusoidal": Transformer-like positional embedding using sin and cos. The wavelength is different from Transformer paper. See implementation

"anchor": Represent the location like YOLO --- The coordinate space is divided into cells, and the coordinate of an object
          is represented as two one-hot vectors indicating the grid and offsets from the center of the each cell.
"""
    B, O, F = pres.shape
    transitions = np.stack([pres,sucs], axis=1) # [B, 2, O, F]
    print(transitions.shape)
    if randomize_locations:
        # add uniform noise of size max(H,W)
        size = max(parameters["picsize"][0][0],
                   parameters["picsize"][0][1])
        transitions = location_augmentation(transitions, size)
        parameters["picsize"][0][0] += size
        parameters["picsize"][0][1] += size

    transitions = {
        None : lambda x: x,
        "random"  : shuffle_objects,
        "match_images_sinkhorn": match_images_sinkhorn,
        "match_images_argmin": match_images_argmin,
    }[object_ordering](transitions)

    insert_patch_shape(F-4)

    return {
        "anchor": anchor_embedding,
        "coord" : coord_embedding,
        "binary": binary_embedding,
        "bbox"  : bbox_embedding,
        "sinusoidal": sinusoidal_embedding,
    }[location_representation](transitions)


def shuffle_objects(transitions):
    B, _, O, F = transitions.shape
    print("shuffling objects (different order in pre/suc)")
    transitions = np.reshape(transitions, (B*2, O, F))
    indices = np.arange(O)
    for i,transition in enumerate(transitions):
        random.shuffle(indices)
        transition = transition[indices]
        transitions[i] = transition
    transitions = np.reshape(transitions, (B, 2, O, F))
    print("shuffling done. shape:", transitions.shape)
    return transitions


# helper function
def show_histogram(thing,msg,**kwargs):
    print(msg)
    hist, bin_edges = np.histogram(thing,**kwargs)
    print()
    print(hist)
    print(bin_edges)
    print()


def compute_pairwise_L2(pres,sucs,tag="initial"):
    B,O,F = pres.shape
    L2_pairwise = ((pres - sucs) ** 2).sum(axis=-1) # [B,O]
    show_histogram(L2_pairwise,f"histogram of {tag} L2 distances between pre and suc. more non-zeros mean many mismatch")
    L2 = ((pres[:,:,None,:] - sucs[:,None,:,:]) ** 2).sum(axis=-1) # [B,O,O]
    show_histogram(L2,f"histogram of {tag} pairwise L2 distances. more non-zeros mean many mismatch")
    print(f"{tag} confusion matrix")
    confusion = np.array([
        np.histogram(np.argmin(L2[:,i,:],axis=1),range=(0,O),bins=O)[0]
        for i in range(O)
    ])
    print(confusion)
    print("total mismatch:",np.sum(confusion)-np.trace(confusion))
    return L2


def match_images_sinkhorn(transitions):
    # Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration
    # Altschuler,Weed,Rigollet
    eps = 1e-4
    B, _, O, F = transitions.shape
    print("reordering objects based on pairwise distances (same order in pre/suc, but not necessarily same order across batch)")
    pres_orig = transitions[:,0,:,:] # [B,O,F]
    sucs_orig = transitions[:,1,:,:] # [B,O,F]

    print("extracting images")
    transitions = transitions[:,:,:,:-4]

    # print("normalizing the input for matching")
    # transitions = normalize(transitions.reshape((B*2*O,F-4)),save=False).reshape((B,2,O,F-4))
    pres = transitions[:,0,:,:] # [B,O,F]
    sucs = transitions[:,1,:,:] # [B,O,F]

    L2 = compute_pairwise_L2(pres,sucs,"initial")

    # eta_adjusted = 4 * np.log(O) / eps
    # eps_adjusted = eps / (8 * L2.max())
    # print(f"adjusted error threshold: {eps_adjusted}")

    exp_neg_L2 = np.exp(- L2)
    show_histogram(exp_neg_L2,"histogram of exp(-L2)")
    exp_neg_L2 = np.maximum(1e-39,exp_neg_L2)
    show_histogram(exp_neg_L2,"histogram of exp(-L2) (clipped)")

    print("computing optimal transport with sinkhorn iteration...")
    # sinkhorn iteration
    for i in range(100):
        if (i%2) == 0:
            r = exp_neg_L2.sum(axis=2,keepdims=True)
            error = ((r - 1)**2).max()
            print(f"max error (row) = {error}")
            if error < eps:
                break
            exp_neg_L2 = exp_neg_L2 / r
        else:
            c = exp_neg_L2.sum(axis=1,keepdims=True)
            error = ((c - 1)**2).max()
            print(f"max error (col) = {error}")
            if error < eps:
                break
            exp_neg_L2 = exp_neg_L2 / c

    show_histogram(exp_neg_L2,"histogram of exp(-L2) after sinkhorn iterations")

    mapping = exp_neg_L2.round() # [B,O,O]
    # double check
    mapping_r = mapping.sum(axis=2)
    mapping_c = mapping.sum(axis=1)
    show_histogram(mapping_r,"histogram of the rounded results (row)")
    show_histogram(mapping_c,"histogram of the rounded results (col)")

    sucs_mapped = np.einsum("bij,bjf->bif",mapping,sucs) # [B,O,F]
    sucs_orig_mapped = np.einsum("bij,bjf->bif",mapping,sucs_orig) # [B,O,F]

    L2 = compute_pairwise_L2(pres,sucs_mapped,"final")

    transitions = np.concatenate([pres_orig,sucs_orig_mapped],axis=1).reshape((B,2,O,F))
    return transitions


def match_images_argmin(transitions):
    # Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration
    # Altschuler,Weed,Rigollet
    eps = 1e-4
    B, _, O, F = transitions.shape
    print("reordering objects based on pairwise distances (same order in pre/suc, but not necessarily same order across batch)")
    pres_orig = transitions[:,0,:,:] # [B,O,F]
    sucs_orig = transitions[:,1,:,:] # [B,O,F]

    print("extracting images")
    transitions = transitions[:,:,:,:-4]

    # print("normalizing the input for matching")
    # transitions = normalize(transitions.reshape((B*2*O,F-4)),save=False).reshape((B,2,O,F-4))
    pres = transitions[:,0,:,:] # [B,O,F]
    sucs = transitions[:,1,:,:] # [B,O,F]

    L2 = compute_pairwise_L2(pres,sucs,"initial") # [B,O,O]

    mapping = np.eye(O)[np.argmin(L2,axis=2)] # [B,O]
    
    sucs_mapped = np.einsum("bij,bjf->bif",mapping,sucs) # [B,O,F]
    sucs_orig_mapped = np.einsum("bij,bjf->bif",mapping,sucs_orig) # [B,O,F]

    L2 = compute_pairwise_L2(pres,sucs_mapped,"final")

    transitions = np.concatenate([pres_orig,sucs_orig_mapped],axis=1).reshape((B,2,O,F))
    import sys
    sys.exit()
    return transitions


def insert_patch_shape(patch_dim):
    # Information inserted here is later used for rendering / visualization
    import math
    # note : the values in parameters are made into a single-element list
    #  so that the tuner does not interpret it as a list of hyperparameters
    picsize = np.array(parameters["picsize"][0])
    if len(picsize) == 3:       # color image
        # always assume a square image patch
        patch_size = int(math.sqrt(patch_dim / 3))
        patch_shape = (patch_size,patch_size,3)
    else:                       # monochrome image
        # always assume a square image patch
        patch_size = int(math.sqrt(patch_dim))
        patch_shape = (patch_size,patch_size)
    parameters["patch_shape"] = [patch_shape]
    parameters["patch_dim"]   = [patch_dim]
    print("patch shape",patch_shape)
    print("patch dim",patch_dim)
    return


def bbox_embedding(transitions):
    B, X, O, F = transitions.shape
    assert X == 2
    transitions = np.reshape(transitions, [B*2*O, F]) # [2BO, F]
    images = transitions[:,:-4]
    print(f"image value range: [{images.min()},{images.max()}]")
    bboxes = transitions[:,-4:]
    print(f"bbox value range: [{bboxes.min()},{bboxes.max()}]")
    transitions = normalize(transitions)
    transitions = transitions.reshape([B, 2, O, F])
    states      = transitions.reshape([B*2,  O, F])
    return transitions, states


def coord_embedding(transitions):
    B, X, O, F = transitions.shape
    assert X == 2
    transitions = np.reshape(transitions, [B*2*O, F]) # [2BO, F]
    images = transitions[:,:-4]
    print(f"image value range: [{images.min()},{images.max()}]")
    bboxes = transitions[:,-4:]
    print(f"bbox value range: [{bboxes.min()},{bboxes.max()}]")
    bboxes = bboxes_to_coord(bboxes)
    transitions = np.concatenate((images, bboxes), axis = -1)
    transitions = normalize(transitions)
    transitions = transitions.reshape([B, 2, O, F])
    states      = transitions.reshape([B*2,  O, F])
    return transitions, states


def sinusoidal_embedding(transitions):
    B, X, O, F = transitions.shape
    assert X == 2
    transitions = np.reshape(transitions, [B*2*O, F]) # [2BO, F]
    images = transitions[:,:-4]
    print(f"image value range: [{images.min()},{images.max()}]")
    bboxes = transitions[:,-4:]
    print(f"bbox value range: [{bboxes.min()},{bboxes.max()}]")
    # note : requires 16 dims (8 sin dims, 2^8 = 256)
    bboxes = bboxes_to_sinusoidal(bboxes,dimensions=16) # 4 * 16 = 64 dims
    # normalize images only
    images  = normalize(images)
    transitions = np.concatenate((images, bboxes), axis=-1)
    transitions = transitions.reshape([B, 2, O, F-4+4*16])
    states      = transitions.reshape([B*2,  O, F-4+4*16])
    return transitions, states


def binary_embedding(transitions):
    B, X, O, F = transitions.shape
    assert X == 2
    transitions = np.reshape(transitions, [B*2*O, F]) # [2BO, F]
    images = transitions[:,:-4]
    print(f"image value range: [{images.min()},{images.max()}]")
    bboxes = transitions[:,-4:]
    print(f"bbox value range: [{bboxes.min()},{bboxes.max()}]")
    bboxes = bboxes_to_binary(bboxes,dimensions=8) # 8x4 = 32 dims
    # normalize images only
    images  = normalize(images)
    transitions = np.concatenate((images, bboxes), axis=-1)
    transitions = transitions.reshape([B, 2, O, F-4+4*8])
    states      = transitions.reshape([B*2,  O, F-4+4*8])
    return transitions, states


def anchor_embedding(transitions):
    B, X, O, F = transitions.shape
    assert X == 2
    transitions = np.reshape(transitions, [B*2*O, F]) # [2BO, F]
    images = transitions[:,:-4]
    print(f"image value range: [{images.min()},{images.max()}]")
    bboxes = transitions[:,-4:]
    print(f"bbox value range: [{bboxes.min()},{bboxes.max()}]")
    cell_size = 8
    anchor_onehot, offsets, grid_size = bboxes_to_anchor(bboxes, parameters["picsize"][0], cell_size) # 8x4 = 32 dims
    parameters["grid_size"] = [grid_size]
    parameters["cell_size"] = [cell_size]
    # normalize images + offsets only
    images_offsets = np.concatenate((images,offsets),axis=-1)
    images_offsets = normalize(images_offsets)
    transitions = np.concatenate((images_offsets, anchor_onehot), axis=-1)
    transitions = transitions.reshape([B, 2, O, -1])
    states      = transitions.reshape([B*2,  O, -1])
    return transitions, states




import numpy as np

def bboxes_to_onehot(bboxes,X,Y):
    batch, objs = bboxes.shape[0:2]

    bboxes_grid   = bboxes // 5
    x1            = bboxes_grid[...,0].flatten()
    y1            = bboxes_grid[...,1].flatten()
    x2            = bboxes_grid[...,2].flatten()
    y2            = bboxes_grid[...,3].flatten()
    x1o           = np.eye(X)[x1].reshape((batch,objs,X))
    y1o           = np.eye(Y)[y1].reshape((batch,objs,Y))
    x2o           = np.eye(X)[x2].reshape((batch,objs,X))
    y2o           = np.eye(Y)[y2].reshape((batch,objs,Y))
    bboxes_onehot = np.concatenate((x1o,y1o,x2o,y2o),axis=-1)
    del x1,y1,x2,y2,x1o,y1o,x2o,y2o
    return bboxes_onehot


def bboxes_to_coord(bboxes):
    coord1, coord2 = bboxes[...,0:2], bboxes[...,2:4]
    center, width = (coord2+coord1)/2, (coord2-coord1)/2
    coords        = np.concatenate((center,width),axis=-1)
    return coords


def coord_to_bboxes(coord):
    cxcy, hw = coord[...,0:2], coord[...,2:4]
    x1y1 = cxcy - hw
    x2y2 = cxcy + hw
    bbox = np.concatenate((x1y1,x2y2),axis=-1)
    return bbox


def bboxes_to_sinusoidal(bboxes,dimensions=16):
    assert (dimensions % 2) == 0
    *shape, F = bboxes.shape
    assert F == 4
    D = dimensions // 2
    k = np.arange(D)

    # w = 0.0001 ** (2 * k / dimensions)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # we don't directly use this formulation in the transformer paper for easier decoding.
    # we wish the unit sine to have width 2
    # 0   1   2
    # _/~\_   _
    #      \_/
    # 
    # k = 1 : 2 -> 2pi
    # k = 2 : 2 ->  pi
    # k = 3 : 2 ->  pi/2
    # therefore, in general,
    import math
    w = (2 * math.pi) * 2.0 ** (-k)
    wt = np.outer(bboxes[...,None],w)
    sin = np.sin(wt)
    cos = np.cos(wt)
    result = np.concatenate((sin,cos),axis=-1)
    result = result.reshape((*shape,F*D*2))
    return result


def bboxes_to_binary(bboxes,dimensions=16):
    # similar to propositional
    assert (dimensions % 2) == 0
    *shape, F = bboxes.shape
    assert F == 4
    D = dimensions
    k = np.arange(1,D+1)
    # D    = 5
    # k    = 1,2,3,4,5
    # 2**k = 2,4,8,16,32
    # bboxes = 9
    # bboxes % (2**k) = 1, 1, 1, 0, 0
    result = bboxes[...,None] % (2 ** k)
    result = result.reshape((*shape,F*D))
    return result


def bboxes_to_anchor(bboxes,picsize,cell_size):
    *shape, F = bboxes.shape
    assert F == 4
    ymax, xmax, *_ = picsize
    # avoid the corner case that points are right on the bottom/right edges
    bboxes = np.clip(bboxes, 0.01, np.array([xmax,ymax,xmax,ymax])-0.01)

    grid_x = xmax // cell_size  # e.g. xmax = 48, cell_size = 16, grid_x = 3
    grid_y = ymax // cell_size

    anchors = (bboxes // cell_size).astype(int) # note: value should be in [0,2] because bboxes < 47.99
    offsets = bboxes % cell_size # no need to subtract the half width, because we will normalize it to mean 0 later

    x1 = anchors[...,0].flatten()
    y1 = anchors[...,1].flatten()
    x2 = anchors[...,2].flatten()
    y2 = anchors[...,3].flatten()
    x1 = np.eye(grid_x)[x1].reshape((*shape,grid_x))
    y1 = np.eye(grid_y)[y1].reshape((*shape,grid_y))
    x2 = np.eye(grid_x)[x2].reshape((*shape,grid_x))
    y2 = np.eye(grid_y)[y2].reshape((*shape,grid_y))

    anchors_onehot = np.concatenate((x1,y1,x2,y2),axis=-1)

    return anchors_onehot, offsets, [grid_x, grid_y]


def binary_to_bboxes(binary):
    shape = binary.shape
    assert (shape[-1] % 4) == 0
    D = shape[-1] // 4
    k = np.arange(1,D+1)
    binary = binary.reshape([*shape[:-1],4,D])
    result = (binary * (2 ** k)).sum(axis=-1)
    result = result.reshape([*shape[:-1], 4])
    return result


def sinusoidal_to_bboxes(sinusoidal):
    binary = sinusoidal >= 0
    return binary_to_bboxes(binary)


def anchor_to_bboxes(anchors_onehot, offsets, grid_size, cell_size):
    grid_x, grid_y = grid_size
    x1 = np.argmax(anchors_onehot[...,:grid_x]                                         ,axis=-1)[...,None]
    y1 = np.argmax(anchors_onehot[...,grid_x:grid_x+grid_y]                            ,axis=-1)[...,None]
    x2 = np.argmax(anchors_onehot[...,grid_x+grid_y:grid_x+grid_y+grid_x]              ,axis=-1)[...,None]
    y2 = np.argmax(anchors_onehot[...,grid_x+grid_y+grid_x:grid_x+grid_y+grid_x+grid_y],axis=-1)[...,None]
    anchors = np.concatenate((x1,y1,x2,y2),axis=-1)
    bboxes = anchors * cell_size + offsets
    return bboxes.astype(int)


def tiled_bboxes(batch, height, width, tilesize):
    x1 = np.tile(np.arange(width),height)   # [9] : 0,1,2, 0,1,2, 0,1,2
    y1 = np.repeat(np.arange(height),width) # [9] : 0,0,0, 1,1,1, 2,2,2
    x2 = x1+1
    y2 = y1+1
    bboxes = \
        np.repeat(                                     # [B,9,4]
            np.expand_dims(                            # [1,9,4]
                np.stack([x1,y1,x2,y2],axis=1) * tilesize, # [9,4]
                0),
            batch, axis=0)
    # [batch, objects, 4]
    return bboxes


def image_to_tiled_objects(x, tilesize):
    B, H, W, C = x.shape
    sH, sW = H//tilesize, W//tilesize

    x = x.reshape([B, sH, tilesize, sW, tilesize, C])
    x = np.swapaxes(x, 2, 3) # [B, sH, sW, tilesize, tilesize, C]
    x = x.reshape([B, sH*sW, tilesize*tilesize*C])
    return x



def random_object_masking(transitions,target_number_of_object=5,threashold=1e-8,augmentation=1):
    """Given a set of transitions, remove the static objects randomly so that the total number of objects match the target.
The algorithm is based on reservoir sampling."""

    B, _, O, F = transitions.shape

    results = [_random_object_masking(transitions,target_number_of_object,threashold)
               for i in range(augmentation)]

    # reorder to avoid data leakage: first 90% remains in the first 90% after the augmentation.
    # [aug_iter, B, 2, O', F] -> [B, aug_iter, 2, O', F] -> [B*aug_iter, 2, O', F]
    return np.swapaxes(np.stack(results, axis=0),0,1).reshape((-1,2,target_number_of_object,F))


def _random_object_masking(transitions,target_number_of_object=5,threashold=1e-8):
    B, _, O, F = transitions.shape

    results = np.zeros((B, 2, target_number_of_object, F))
    changed_item_count = np.zeros(B,dtype=np.int8)

    pres = transitions[:,0]     # B,O,F
    sucs = transitions[:,1]     # B,O,F

    diff = abs(pres - sucs)
    same   = np.all(diff <= threashold, axis=2) # B,O
    changed = np.logical_not(same)              # B,O

    # copy changed objects first
    changed_idxs = np.where(changed)
    for b, o in zip(*changed_idxs):
        count = changed_item_count[b]
        results[b,:,count] = transitions[b,:,o]
        changed_item_count[b] = count+1

    reservoir_size = target_number_of_object - changed_item_count # [B]
    assert np.all(reservoir_size >= 0)
    reservoir_count = np.zeros(B,dtype=np.int8) # [B]

    # copy unchanged objects randomly, with reservoir sampling
    import random
    same_idxs = np.where(same)
    for b, o in zip(*same_idxs):
        size   = reservoir_size[b]
        count  = reservoir_count[b]
        offset = changed_item_count[b]
        if count < size:
            # fill the reservoir array.
            results[b,:,offset+count] = transitions[b,:,o]
        else:
            j = random.randint(0, count)
            if j < size:
                results[b,:,offset+j] = transitions[b,:,o]
        reservoir_count[b] = count+1

    return results



def location_augmentation(transitions,height=100,width=100,augmentation=1):
    results = [_location_augmentation(transitions,height,width)
               for i in range(int(augmentation))]
    # reorder to avoid data leakage: first 90% remains in the first 90% after the augmentation.
    # [aug_iter, B, 2, O', F] -> [B, aug_iter, 2, O', F] -> [B*aug_iter, 2, O', F]
    return np.swapaxes(np.stack(results, axis=0),0,1).reshape((-1,*transitions.shape[1:]))


def _location_augmentation(transitions,height,width):
    B,_,O,F = transitions.shape
    x_noise = np.random.uniform(low=0.0,high=width,size=(B,1,1)).astype(transitions.dtype)
    y_noise = np.random.uniform(low=0.0,high=height,size=(B,1,1)).astype(transitions.dtype)
    transitions = transitions.copy()
    transitions[...,-4] += x_noise
    transitions[...,-3] += y_noise
    transitions[...,-2] += x_noise
    transitions[...,-1] += y_noise
    return transitions

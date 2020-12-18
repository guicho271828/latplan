
import numpy as np

def bboxes_to_onehot(bboxes,X,Y):
    batch, objs = bboxes.shape[0:2]

    bboxes_grid   = bboxes // 5
    x1            = bboxes_grid[:,:,0].flatten()
    y1            = bboxes_grid[:,:,1].flatten()
    x2            = bboxes_grid[:,:,2].flatten()
    y2            = bboxes_grid[:,:,3].flatten()
    x1o           = np.eye(X)[x1].reshape((batch,objs,X))
    y1o           = np.eye(Y)[y1].reshape((batch,objs,Y))
    x2o           = np.eye(X)[x2].reshape((batch,objs,X))
    y2o           = np.eye(Y)[y2].reshape((batch,objs,Y))
    bboxes_onehot = np.concatenate((x1o,y1o,x2o,y2o),axis=-1)
    del x1,y1,x2,y2,x1o,y1o,x2o,y2o
    return bboxes_onehot


def bboxes_to_coord(bboxes):
    coord1, coord2 = bboxes[:,:,0:2], bboxes[:,:,2:4]
    center, width = (coord2+coord1)/2, (coord2-coord1)/2
    coords        = np.concatenate((center,width),axis=-1)
    return coords


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
    ""

    B, _, O, F = transitions.shape

    results = [_random_object_masking(transitions,target_number_of_object,threashold)
               for i in range((O//target_number_of_object) * (int(augmentation)))]

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



def location_augmentation(transitions,height=100,width=100,augmentation=1,mode="coord"):
    assert mode in ["coord", "bbox"]
    results = [_location_augmentation(transitions,height,width,mode)
               for i in range(int(augmentation))]
    # reorder to avoid data leakage: first 90% remains in the first 90% after the augmentation.
    # [aug_iter, B, 2, O', F] -> [B, aug_iter, 2, O', F] -> [B*aug_iter, 2, O', F]
    return np.swapaxes(np.stack(results, axis=0),0,1).reshape((-1,*transitions.shape[1:]))


def _location_augmentation(transitions,height,width,mode):
    B,_,O,F = transitions.shape
    x_noise = np.random.uniform(low=0.0,high=width,size=(B,1,1)).astype(transitions.dtype)
    y_noise = np.random.uniform(low=0.0,high=height,size=(B,1,1)).astype(transitions.dtype)
    transitions = transitions.copy()
    transitions[:,:,:,-4] += x_noise
    transitions[:,:,:,-3] += y_noise
    if mode == "bbox":
        transitions[:,:,:,-2] += x_noise
        transitions[:,:,:,-1] += y_noise
    return transitions


import numpy as np

# the code is slow --- do not run these functions on a large number of states/transitions!

# It quantifies the image into 3-bit colors,
# then count the number of pixels to detect the objects.
# the most frequent color (gray) is ignored as the background.
# also, colors which occur less than 0.1% of the screen are ignored.
# Finally, for each color, we compute the centroid and width/height to obtain the estimate of the object location.

# We check the validity of the state as follows:
# If an object is a "bottom" object (nothing is below), it compares the height with other bottom objects
# and ensure that it is not in the mid air.
# Otherwise, it searches for objects below it, and see one object is directly below it.
# 
# We check the validity of the transitions as follows:
# We first look for unaffected objects, and check exactly one object is moved. (no more, no less)
# The moved object should be a "top" objects both before and after the transition.

# Note:
# With cylinders dataset in 150x100 resolution and table size 5,
# the horizontal center-center distance is about 18 pixels : about 12% --- half distance is 6%
# the vertical center-center distance is about 14 pixels   : about 14% --- half distance is 7%.

def setup():
    pass


def quantize(images):
    return (((images*2).round()/2)*256).astype(int)


def remove_outliers(vector):
    q1 = np.quantile(vector,0.25)
    q3 = np.quantile(vector,0.75)
    iqr = q3 - q1
    lb = q1 - iqr * 1.5
    ub = q3 + iqr * 1.5
    return np.logical_and(lb <= vector, vector <= ub)


from dataclasses import dataclass

@dataclass
class block:
    x: float                    # x coordinate
    y: float                    # y coordinate
    cw: float                   # HALF the width of the object
    ch: float                   # HALF the height of the object
    def same_tower(o1, o2):
        return (abs(o1.x - o2.x) < (o1.cw + o2.cw)/2)

    def same_level(o1, o2):
        return (abs(o1.y - o2.y) < (o1.ch + o2.ch)/2)

    def same_bottom(o1, o2):
        # in addition to same_level , adds the height to the center coordinate and obtain the bottom height
        # Note: larger y value == toward the bottom of the image
        return (abs((o1.y+o1.ch) - (o2.y+o2.ch)) < (o1.ch + o2.ch)/2)

    def similar(o1, o2):
        return o1.same_tower(o2) and o1.same_level(o2)

    def above(o1, o2):
        return o1.same_tower(o2) and (o1.y < o2.y) # Note: larger y value == toward the bottom of the image

    def on(o1, o2):
        # y difference must be around the sum of both heights.
        return o1.above(o2) and \
            abs(o1.y - o2.y) < (o1.ch + o2.ch) * 1.5 and \
            abs(o1.y - o2.y) > (o1.ch + o2.ch) * 0.5



# find top objects
def tops(objects):
    results = []
    for o1 in objects:
        top = True
        for o2 in objects:
            if o1 == o2:
                continue
            if o2.above(o1):
                top = False
                break
        if top:
            results.append(o1)
    return results


# find bottom objects
def bottoms(objects):
    results = []
    for o1 in objects:
        bottom = True
        for o2 in objects:
            if o1 == o2:
                continue
            if o1.above(o2):
                bottom = False
                break
        if bottom:
            results.append(o1)
    return results


# parse a scene
def to_config(state, verbose=False):
    # quantilze the color: each RGB value would be one of 0,128,256
    quantized = quantize(state)

    # count the number of pixels for each quantized color
    colors = quantized.reshape((-1, 3))
    from collections import Counter
    counter = Counter(list(map(tuple,colors)))

    # extract the colors for objects.
    # we first sort the colors based on frequencies
    colorlist = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    # we ignore the most frequent color which is the background
    colorlist.pop(0)
    # we ignore pixels that are less than 0.1 % as noise
    threshold = len(colors) * 0.005
    object_colors = [
        color for color, count in colorlist
        if count > threshold
    ]
    # if verbose:
    #     print(f"{len(object_colors)} colors: {object_colors}")

    # obtain the object locations.
    # collect the pixels that match the color, then find the centroid
    blocks = []
    for color in object_colors:
        ys, xs = np.where(np.all(quantized == color, axis=-1))

        # remove outliers
        keep = np.where(np.logical_and(remove_outliers(ys),remove_outliers(xs))) 
        # if verbose:
        #     print(ys,"->")
        ys = ys[keep]
        # if verbose:
        #     print(ys)
        #     print(xs,"->")
        xs = xs[keep]
        # if verbose:
        #     print(xs)
        
        y, x = float(ys.mean()), float(xs.mean())
        h, w = float(ys.max()-ys.min())/2, float(xs.max()-xs.min())/2
        blocks.append(block(x, y, w, h))

    return blocks


# check if towers are properly stacked
def validate_state(blocks,verbose):
    results = []
    # there are no absolute measure for the objects on the table.
    # we check if they have the similar height
    bottom_objects = bottoms(blocks)
    for o1 in bottom_objects:
        for o2 in bottom_objects:
            if not o1.same_bottom(o2):
                if verbose:
                    print(f"all bottom objects must have the same bottom height: {bottom_objects}")
                return False

    for o1 in blocks:
        if o1 not in bottom_objects:
            # this is not an object on the table,
            # therefore it must be on something else.
            objs_below = [ o2 for o2 in blocks if o1.above(o2) ]
            highest_object_below_o1 = tops(objs_below)
            assert len(highest_object_below_o1) == 1
            o2 = highest_object_below_o1[0]
            if not o1.on(o2):
                if verbose:
                    print(f"object o1 is hovering over o2: o1={o1} o2={o2}")
                return False
    return True


def validate_states(states, verbose=True, **kwargs):

    results = np.zeros(len(states),dtype=bool)

    for i,state in enumerate(states):
        blocks = to_config(state,verbose=verbose)
        results[i] = validate_state(blocks,verbose)
    
    return results


def validate_transitions(transitions, check_states=True, verbose=True, **kwargs):
    pres = transitions[0]
    sucs = transitions[1]
    results = np.zeros(len(pres),dtype=bool)

    for i, (pre, suc) in enumerate(zip(pres, sucs)):
        pre_blocks = to_config(pre,verbose=verbose)
        suc_blocks = to_config(suc,verbose=verbose)
        if check_states:
            if not validate_state(pre_blocks,verbose=verbose):
                continue
            if not validate_state(suc_blocks,verbose=verbose):
                continue

        matches     = []
        moved_1     = pre_blocks.copy()
        moved_2     = suc_blocks.copy()
        not_moved_1 = []
        not_moved_2 = []
        for o1 in pre_blocks:
            for o2 in suc_blocks:
                if o1.similar(o2):
                    # matching object found!
                    matches.append((o1,o2))
                    not_moved_1.append(o1)
                    not_moved_2.append(o2)
                    moved_1.remove(o1)
                    moved_2.remove(o2)
                    break

        if len(moved_1) != 1 or len(moved_2) != 1:
            if verbose:
                print(f"exactly one object should be moved: {moved_1}, {moved_2}")
            continue

        moved_1 = moved_1[0]
        moved_2 = moved_2[0]
        
        if (moved_1 not in tops(pre_blocks)):
            if verbose:
                print(f"moved object in previous state should be top objects: {moved_1}, {tops(pre_blocks)}")
            continue
        if (moved_2 not in tops(suc_blocks)):
            if verbose:
                print(f"moved object in successor state should be top objects: {moved_2}, {tops(suc_blocks)}")
            continue

        results[i] = True
        
    return results

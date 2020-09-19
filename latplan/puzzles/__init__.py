
from . import model

from . import counter_mnist
from . import counter_random_mnist
from . import lightsout_digital
from . import lightsout_twisted
from . import hanoi
from . import puzzle_digital
from . import puzzle_lenna
from . import puzzle_mandrill
from . import puzzle_mnist
from . import puzzle_wrong
from . import puzzle_spider
from . import split_image

def shuffle_objects(objects):
    import numpy as np
    import numpy.random
    tmp = np.copy(objects)
    if len(tmp.shape) == 4:
        # shuffling transitions
        assert tmp.shape[1] == 2
        # B, 2, O, F
        tmp = np.swapaxes(tmp, 0, 2)
        np.random.shuffle(tmp)
        tmp = np.swapaxes(tmp, 0, 2)
    else:
        # shuffling states
        assert len(tmp.shape) == 3
        # B, O, F
        tmp = np.swapaxes(tmp, 0, 1)
        np.random.shuffle(tmp)
        tmp = np.swapaxes(tmp, 0, 1)
    return tmp

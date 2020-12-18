
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

from . import objutil

def shuffle_objects(x, copy=True):
    import numpy as np
    import numpy.random
    if copy:
        x = np.copy(x)
    if len(x.shape) == 4:
        # shuffling transitions
        assert x.shape[1] == 2
        # B, 2, O, F
        x = np.swapaxes(x, 0, 2)
        np.random.shuffle(x)
        x = np.swapaxes(x, 0, 2)
    else:
        # shuffling states
        assert len(x.shape) == 3
        # B, O, F
        x = np.swapaxes(x, 0, 1)
        np.random.shuffle(x)
        x = np.swapaxes(x, 0, 1)
    return x

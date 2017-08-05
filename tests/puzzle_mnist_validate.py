#!/usr/bin/env python3

import importlib
import numpy as np

import latplan
import latplan.puzzles.puzzle_mnist as p
import latplan.puzzles.model.puzzle as m

importlib.reload(m)
importlib.reload(p)

p.setup()


import itertools
c = [ c for c in itertools.islice(p.generate_configs(9), 10000) ]
s = p.states(3,3,c)

from latplan.util.timer import Timer
with Timer("************************* validate_states on cpu ***************************"):
    assert np.all(m.validate_states_cpu(s,3,3))

with Timer("************************* validate_states on gpu, batch=100 ***************************"):
    assert np.all(m.validate_states_gpu(s,3,3,batch_size=100))

with Timer("************************* validate_states on gpu, batch=1000 ***************************"):
    assert np.all(m.validate_states_gpu(s,3,3,batch_size=1000))



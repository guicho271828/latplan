#!/usr/bin/env python3

import importlib
import numpy as np

import latplan
import latplan.puzzles.puzzle_mnist as p

importlib.reload(p)

p.setup()


import itertools
c = [ c for c in itertools.islice(p.generate_configs(9), 10000) ]

from colors import color
from functools import partial

style = partial(color, fg='black', bg='white')

from latplan.util.timer import Timer

with Timer(style("************************* states on cpu ***************************")):
    s = p.generate_cpu(c,3,3)

with Timer(style("************************* states on gpu, batch=100 ***************************")):
    s = p.generate_gpu(c,3,3, batch_size=100)

with Timer(style("************************* states on gpu, batch=1000 ***************************")):
    s = p.generate_gpu(c,3,3, batch_size=1000)


with Timer(style("************************* validate_states on cpu ***************************")):
    print("results:", np.all(p.validate_states_cpu(s)), "(should be True)")

with Timer(style("************************* validate_states on gpu, batch=100 ***************************")):
    print("results:", np.all(p.validate_states_gpu(s,batch_size=100)), "(should be True)")

with Timer(style("************************* validate_states on gpu, batch=1000 ***************************")):
    print("results:", np.all(p.validate_states_gpu(s,batch_size=1000)), "(should be True)")

with Timer(style("************************* to_configs on cpu ***************************")):
    print(p.to_configs_cpu(s)[:3])

with Timer(style("************************* to_configs on gpu, batch=100 ***************************")):
    print(p.to_configs_gpu(s,batch_size=100)[:3])

with Timer(style("************************* to_configs on gpu, batch=1000 ***************************")):
    print(p.to_configs_gpu(s,batch_size=1000)[:3])

c = c[:10]

with Timer(style("************************* transitions_old ***************************")):
    transitions = p.transitions_old(3,3,configs=c)

with Timer(style("************************* transitions ***************************")):
    transitions = p.transitions(3,3,configs=c)

with Timer(style("************************* transitions_old one_per_state ***************************")):
    transitions = p.transitions_old(3,3,configs=c,one_per_state=True)

with Timer(style("************************* transitions one_per_state ***************************")):
    transitions = p.transitions(3,3,configs=c,one_per_state=True)


with Timer(style("************************* validate_transitions_cpu_old ***************************")):
    print(p.validate_transitions_cpu_old(transitions,batch_size=1000)[:3])

with Timer(style("************************* validate_transitions_cpu ***************************")):
    print(p.validate_transitions_cpu(transitions,batch_size=1000)[:3])


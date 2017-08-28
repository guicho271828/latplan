#!/usr/bin/env python3

import importlib
import numpy as np

import latplan
import latplan.puzzles.lightsout_digital as p

# importlib.reload(p)

import itertools
c = [ c for c in itertools.islice(p.generate_configs(3), 10000) ]

from colors import color
from functools import partial

style = partial(color, fg='black', bg='white')

from latplan.util.timer import Timer

with Timer(style("************************* states on cpu ***************************")):
    s = p.generate_cpu(c)
    print(s[120])

with Timer(style("************************* states on gpu, batch=100  ***************************")):
    s = p.generate_gpu(c, batch_size=100)
    print(s[120])

with Timer(style("************************* states on gpu, batch=1000 ***************************")):
    s = p.generate_gpu(c, batch_size=1000)
    print(s[120])

with Timer(style("************************* validate_states on cpu ***************************")):
    print("results:", np.all(p.validate_states(s)), "(should be True)")

with Timer(style("************************* validate_states with noise ***************************")):
    print("results:", np.all(p.validate_states(np.clip(s+np.random.normal(0,0.1,s.shape),0,1))), "(should be True)")


with Timer(style("************************* to_configs on gpu, batch=100 ***************************")):
    p.to_configs(s,batch_size=100)
    

with Timer(style("************************* to_configs on gpu, batch=1000 ***************************")):
    p.to_configs(s,batch_size=1000)

print("config - to_configs(generate(config)):", np.sum(np.abs(c - p.to_configs(s,batch_size=1000))))
print("original     :",c[120])
print("reconstructed:",p.to_configs(s,batch_size=100)[120])

c = c[:10]

with Timer(style("************************* transitions ***************************")):
    transitions = p.transitions(3,configs=c)

with Timer(style("************************* transitions one_per_state ***************************")):
    transitions = p.transitions(3,configs=c,one_per_state=True)

with Timer(style("************************* validate_transitions_cpu ***************************")):
    print("all transitions valid?:",np.all(p.validate_transitions(transitions,batch_size=1000)))


#!/usr/bin/env python3

import importlib
import numpy as np

import latplan
import latplan.puzzles.lightsout_twisted as p

# importlib.reload(p)

import itertools
c = np.array([ c for c in itertools.islice(p.generate_configs(4), 1000) ])

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

for i in range(3,5):
    p.threshold = i*0.01
    with Timer(style("************************* validate_states with threshold = {} ***************************".format(p.threshold))):
        results = p.validate_states(s,batch_size=1000)
    print("results:", np.all(results), "(should be True)")
    print("how many invalid? : ", len(results)-np.count_nonzero(results), "/", len(results))



with Timer(style("************************* to_configs on gpu, batch=100 ***************************")):
    p.to_configs(s,batch_size=100)
    

with Timer(style("************************* to_configs on gpu, batch=1000 ***************************")):
    p.to_configs(s,batch_size=1000)

from latplan.util import bce, mae
_c = p.to_configs(s,batch_size=1000)
print("sum(abs(config - to_configs(generate(config)))) =", np.sum(np.abs(c - _c)))
print("sum(abs(config - round(to_configs(generate(config))))) =", np.sum(np.abs(c - _c.round())))
print("MAE(config, to_configs(generate(config))) =", mae(c, _c))
print("MAE(config, round(to_configs(generate(config)))) =", mae(c, _c.round()))
print("BCE(config, to_configs(generate(config))) =", bce(c, _c))
print("BCE(config, round(to_configs(generate(config)))) =", bce(c, _c.round()))
for i in range(120,125):
    print("original     :",c[i])
    print("reconstructed:",_c[i])
    

c = c[:10]

with Timer(style("************************* transitions_old ***************************")):
    transitions = p.transitions_old(4,configs=c)

with Timer(style("************************* transitions ***************************")):
    transitions = p.transitions(4,configs=c)

with Timer(style("************************* transitions one_per_state ***************************")):
    transitions = p.transitions(4,configs=c,one_per_state=True)

with Timer(style("************************* validate_transitions_cpu ***************************")):
    results = p.validate_transitions(transitions,batch_size=1000)

print("all transitions valid?:",np.all(results))
print("if not, how many invalid?:",len(results)-np.count_nonzero(results), "/", len(results))


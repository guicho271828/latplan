#!/usr/bin/env python3

import numpy as np
import sys
sys.path.append('../../')

import latplan.puzzles.lightsout_twisted as p

from plot import plot_image, plot_grid

configs = p.generate_configs(4)
puzzles = p.generate_gpu(configs[-30:])
print(puzzles[-1])
plot_image(puzzles[-1],"lightsout_twisted_gpu.png")
puzzles = p.generate_cpu(configs[-30:])
print(puzzles[-1])
plot_image(puzzles[-1],"lightsout_twisted_cpu.png")
plot_image(puzzles[-1].round(),"lightsout_twisted_cpu_r.png")
plot_image(p.batch_unswirl(puzzles)[-1],"lightsout_twisted_untwisted.png")
plot_image(p.batch_unswirl(puzzles)[-1].round(),"lightsout_twisted_untwisted_r.png")

_transitions = p.transitions(4, configs=configs[:100], one_per_state=True)
import numpy.random as random
indices = random.randint(0,_transitions[0].shape[0],18)
_transitions = _transitions[:,indices]
print(_transitions.shape)
transitions_for_show = \
    np.einsum('ba...->ab...',_transitions) \
      .reshape((-1,)+_transitions.shape[2:])
print(transitions_for_show.shape)
plot_grid(transitions_for_show,"lightsout_twisted_transitions.png")

unswirled = p.batch_unswirl(transitions_for_show)
plot_grid(unswirled,"lightsout_twisted_transitions_untwisted.png")
plot_grid(unswirled.round(),"lightsout_twisted_transitions_untwisted_rounded.png")

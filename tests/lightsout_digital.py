#!/usr/bin/env python3

import numpy as np
import sys
sys.path.append('../../')

from latplan.puzzles.lightsout_digital import generate_configs, successors, generate, states, transitions

from plot import plot_image, plot_grid

configs = generate_configs(3)
puzzles = generate(configs)
print(puzzles[10])
plot_image(puzzles[10],"lightsout_digital.png")
_transitions = transitions(3)
import numpy.random as random
indices = random.randint(0,_transitions[0].shape[0],18)
_transitions = _transitions[:,indices]
print(_transitions.shape)
transitions_for_show = \
    np.einsum('ba...->ab...',_transitions) \
      .reshape((-1,)+_transitions.shape[2:])
print(transitions_for_show.shape)
plot_grid(transitions_for_show,"lightsout_digital_transitions.png")

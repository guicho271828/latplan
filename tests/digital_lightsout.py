#!/usr/bin/env python3

import numpy as np
import sys
sys.path.append('../../')

from latplan.puzzles.digital_lightsout import generate_configs, successors, generate, states, transitions

from plot import plot_image, plot_grid

configs = generate_configs(3)
puzzles = generate(configs)
print(puzzles[10])
plot_image(puzzles[10],"digital_lightsout.png")
plot_grid(puzzles[:36],"digital_lightsouts.png")
_transitions = transitions(3)
import numpy.random as random
indices = random.randint(0,_transitions[0].shape[0],18)
_transitions = _transitions[:,indices]
print(_transitions.shape)
transitions_for_show = \
    np.einsum('ba...->ab...',_transitions) \
      .reshape((-1,)+_transitions.shape[2:])
print(transitions_for_show.shape)
plot_grid(transitions_for_show,"digital_lightsout_transitions.png")

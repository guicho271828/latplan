#!/usr/bin/env python3

import numpy as np
import sys
sys.path.append('../../')

from plot import puzzle_plot
from latplan.puzzles.puzzle_random_mnist import generate_configs, successors, generate, states, transitions, random_panels

from plot import plot_image, plot_grid

configs = generate_configs(6)
puzzles = generate(configs, 2, 3, random_panels())
print(puzzles[10])
plot_image(puzzles[10],"puzzle_random_mnist.png")
plot_grid(puzzles[:36],"puzzle_random_mnists.png")
_transitions = transitions(2,3)
import numpy.random as random
indices = random.randint(0,_transitions[0].shape[0],18)
_transitions = _transitions[:,indices]
print(_transitions.shape)
transitions_for_show = \
    np.einsum('ba...->ab...',_transitions) \
      .reshape((-1,)+_transitions.shape[2:])
print(transitions_for_show.shape)
plot_grid(transitions_for_show,"puzzle_random_mnist_transitions.png")


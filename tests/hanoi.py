#!/usr/bin/env python3

import numpy as np
import sys
sys.path.append('../../')

from latplan.puzzles.hanoi import generate_configs, successors, generate, states, transitions

from plot import plot_image, plot_grid

disks = 8
towers = 3

configs = generate_configs(disks,towers)
puzzles = generate(configs,disks,towers)
print(puzzles.shape)
print(puzzles[10])

for line in puzzles[10]:
    print(line)

plot_image(puzzles[0],"hanoi.png")
plot_image(np.clip(puzzles[0]+np.random.normal(0,0.1,puzzles[0].shape),0,1),"hanoi+noise.png")
plot_image(np.round(np.clip(puzzles[0]+np.random.normal(0,0.1,puzzles[0].shape),0,1)),"hanoi+noise+round.png")
plot_grid(puzzles[:36],"hanois.png")
_transitions = transitions(disks,towers)
print(_transitions.shape)
import numpy.random as random
indices = random.randint(0,_transitions[0].shape[0],18)
_transitions = _transitions[:,indices]
print(_transitions.shape)
transitions_for_show = \
    np.einsum('ba...->ab...',_transitions) \
      .reshape((-1,)+_transitions.shape[2:])
print(transitions_for_show.shape)
plot_grid(transitions_for_show,"hanoi_transitions.png")


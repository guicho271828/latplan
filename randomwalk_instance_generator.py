#!/usr/bin/env python3


import numpy as np
import numpy.random as random

float_formatter = lambda x: "%.5f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def random_walk(init_c,length,successor_fn):
    current = init_c
    trace = [current]
    for i in range(length):
        sucs = successor_fn(current)
        suc = sucs[random.randint(len(sucs))]
        while suc in trace:
            suc = sucs[random.randint(len(sucs))]
        trace.append(suc)
        current = suc
    return current

def puzzle_mnist(steps, N):
    import latplan.puzzles.puzzle_mnist as p
    for i in range(N):
        print(random_walk([0,1,2,3,4,5,6,7,8], steps, lambda config: p.successors(config,3,3)))

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("Usage : ", sys.argv[0], puzzle_["mnist"], "steps", "N")
    else:
        print('args:',sys.argv)
        task = sys.argv[1]
        globals()[task](*(map(eval,sys.argv[2:])))

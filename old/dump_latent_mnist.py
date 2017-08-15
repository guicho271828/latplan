#!/usr/bin/env python3

import numpy as np
from model import GumbelAE, GumbelAE2, ConvolutionalGumbelAE
from plot import plot_grid, plot_grid2
import puzzles.mnist_puzzle as p

def run(ae):
    ae.load()
    n = ae.parameters['N']
    for i in range(9):
        configs = 11-np.identity(9)*(11-i)
        print(configs)
        ae.plot(p.generate(configs,3,3),
                "single-digit-{}.png".format(i))

# run(GumbelAE2("samples/lightsout_digital_4_fc2"))
# run(GumbelAE( "samples/lightsout_twisted_3_fc"))
# run(GumbelAE2("samples/hanoi3_fc2"))
# run(GumbelAE2("samples/hanoi4_fc2"))
# run(GumbelAE2("samples/mandrill_puzzle33_fc2"))
run(GumbelAE2("samples/mnist_puzzle33_fc2"))
# run(GumbelAE2("samples/xhanoi4_fc2"))

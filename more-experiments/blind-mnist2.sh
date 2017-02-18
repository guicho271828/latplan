#!/bin/bash
./plan.py blind 'run_puzzle("samples/mnist_puzzle33p_fc2","fc2",import_module("puzzles.mnist_puzzle"),init=2)' |& tee $(dirname $0)/blind-mnist2.log

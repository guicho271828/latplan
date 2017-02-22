#!/bin/bash
./plan.py zopdb 'run_puzzle("samples/mnist_puzzle33_fc2","fc2",import_module("puzzles.mnist_puzzle"),init=2)' |& tee $(dirname $0)/zopdb-mnist2.log

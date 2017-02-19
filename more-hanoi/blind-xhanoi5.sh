#!/bin/bash
./plan.py blind 'run_hanoi("samples/xhanoi5_fc2/","fc2",import_module("puzzles.hanoi"),5)' |& tee $(dirname $0)/blind-xhanoi5.log

#!/bin/bash
./plan.py blind 'run_hanoi("samples/xhanoi6_fc2/","fc2",import_module("puzzles.hanoi"),6)' |& tee $(dirname $0)/blind-xhanoi6.log

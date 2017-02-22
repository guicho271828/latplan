#!/bin/bash
./plan.py blind 'run_hanoi("samples/xhanoi8_fc2/","fc2",import_module("puzzles.hanoi"),8)' |& tee $(dirname $0)/blind-xhanoi8.log

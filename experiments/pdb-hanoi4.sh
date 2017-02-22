#!/bin/bash
./plan.py pdb 'run_hanoi ( "samples/hanoi4_fc2" ,"fc2", import_module("puzzles.hanoi" ), 4)' |& tee $(dirname $0)/pdb-hanoi4.log

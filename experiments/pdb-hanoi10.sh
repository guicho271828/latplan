#!/bin/bash
./plan.py pdb 'run_hanoi10   ( "samples/hanoi10_fc2"                 ,"fc2", import_module("puzzles.hanoi"                    ) )' |& tee $(dirname $0)/pdb-hanoi10.log

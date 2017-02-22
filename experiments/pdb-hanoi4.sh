#!/bin/bash
./plan.py pdb 'run_hanoi4    ( "samples/hanoi4_fc2"                  ,"fc2", import_module("puzzles.hanoi"                    ) )' |& tee $(dirname $0)/pdb-hanoi4.log

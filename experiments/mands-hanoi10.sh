#!/bin/bash
./plan.py mands 'run_hanoi10   ( "samples/hanoi10_fc2"                 ,"fc2", import_module("puzzles.hanoi"                    ) )' |& tee $(dirname $0)/mands-hanoi10.log

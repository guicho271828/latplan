#!/bin/bash
./plan.py pdb 'run_lightsout ( "samples/lightsout_digital_4_fc2" ,"fc2", import_module("puzzles.lightsout_digital" ) )' |& tee $(dirname $0)/pdb-lightsout.log

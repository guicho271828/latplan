#!/bin/bash
./plan.py mands 'run_lightsout ( "samples/lightsout_digital_4_fc2" ,"fc2", import_module("puzzles.lightsout_digital" ) )' |& tee $(dirname $0)/mands-lightsout.log

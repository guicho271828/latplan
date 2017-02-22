#!/bin/bash
./plan.py blind 'run_lightsout ( "samples/digital_lightsout_fc2" ,"fc2", import_module("puzzles.digital_lightsout" ) )' |& tee $(dirname $0)/blind-lightsout.log

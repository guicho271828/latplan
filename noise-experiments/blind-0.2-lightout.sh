#!/bin/bash
./plan_noise.py blind 0.2 'run_lightsout ( "samples/digital_lightsout_fc2"       ,"fc2", import_module("puzzles.digital_lightsout"        ) )' |& tee $(dirname $0)/blind-0.2-lightout.log

#!/bin/bash
./plan_noise.py blind 0.08 salt 'run_lightsout3 ( "samples/digital_lightsout_skewed_3_fc"       ,"fc", import_module("puzzles.digital_lightsout_skewed"        ) )' |& tee $(dirname $0)/blind-salt-lightout-skewed3-0.08.log

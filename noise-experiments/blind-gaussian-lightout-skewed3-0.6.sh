#!/bin/bash
./plan_noise.py blind 0.6 gaussian 'run_lightsout3 ( "samples/digital_lightsout_skewed_3_fc"       ,"fc", import_module("puzzles.digital_lightsout_skewed"        ) )' |& tee $(dirname $0)/blind-gaussian-lightout-skewed3-0.6.log

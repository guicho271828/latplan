#!/bin/bash
./plan_noise.py blind 0.2 gaussian 'run_lightsout3 ( "samples/lightsout_twisted_3_fc"       ,"fc", import_module("puzzles.lightsout_twisted"        ) )' |& tee $(dirname $0)/blind-gaussian-lightout-twisted3-0.2.log

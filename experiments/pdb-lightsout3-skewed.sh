#!/bin/bash
./plan.py pdb 'run_lightsout3 ( "samples/digital_lightsout_skewed_3_fc" ,"fc", import_module("puzzles.digital_lightsout_skewed" ) )' |& tee $(dirname $0)/pdb-lightsout-skewed3.log

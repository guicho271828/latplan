#!/bin/bash

trap exit SIGINT

parallel ./plan.py ::: blind pdb \
         ::: \
         'run_hanoi     ( "samples/hanoi_fc2"                   ,"fc2", import_module("puzzles.hanoi")                    )' \
         'run_puzzle    ( "samples/mnist_puzzle33p_fc2"         ,"fc2", import_module("puzzles.mnist_puzzle")             )' \
         'run_puzzle    ( "samples/lenna_puzzle33p_fc2"         ,"fc2", import_module("puzzles.lenna_puzzle")             )' \
         'run_puzzle    ( "samples/mandrill_puzzle33p_fc2"      ,"fc2", import_module("puzzles.mandrill_puzzle")          )' \
         'run_lightsout ( "samples/digital_lightsout_fc2"       ,"fc2", import_module("puzzles.digital_lightsout")        )' \
         # 'run_lightsout ( "samples/digital_lightsout_skewed_fc" ,"fc", import_module("puzzles.digital_lightsout_skewed") )'














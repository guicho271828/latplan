#!/bin/bash

ulimit -v 16000000000

trap exit SIGINT

parallel --eta --timeout 180 --joblog problem-instances/latplan.puzzles.puzzle_mnist.csv  \
         "test -e {2}/*.valid || ./trivial-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.log" \
         ::: puzzle_mnist_3_3_36_20000_conv \
         ::: problem-instances/*/latplan.puzzles.puzzle_mnist/* \
         ::: Astar

parallel --eta --timeout 180 --joblog problem-instances/latplan.puzzles.puzzle_mandrill.csv   \
         "test -e {2}/*.valid || ./trivial-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.log" \
         ::: puzzle_mandrill_3_3_36_20000_conv \
         ::: problem-instances/*/latplan.puzzles.puzzle_mandrill/* \
         ::: Astar


parallel --eta --timeout 180 --joblog problem-instances/latplan.puzzles.puzzle_spider.csv   \
         "test -e {2}/*.valid || ./trivial-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.log" \
         ::: puzzle_spider_3_3_36_20000_conv \
         ::: problem-instances/*/latplan.puzzles.puzzle_spider/* \
         ::: Astar


parallel --eta --timeout 180 --joblog problem-instances/latplan.puzzles.lightsout_digital.csv   \
         "test -e {2}/*.valid || ./trivial-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.log" \
         ::: lightsout_digital_4_36_20000_conv \
         ::: problem-instances/*/latplan.puzzles.lightsout_digital/* \
         ::: Astar


parallel --eta --timeout 180 --joblog problem-instances/latplan.puzzles.lightsout_twisted.csv   \
         "test -e {2}/*.valid || ./trivial-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.log" \
         ::: lightsout_twisted_4_36_20000_conv \
         ::: problem-instances/*/latplan.puzzles.lightsout_twisted/* \
         ::: Astar


parallel --eta --timeout 180 --joblog problem-instances/latplan.puzzles.hanoi.csv   \
         "test -e {2}/*.valid || ./trivial-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.log" \
         ::: hanoi_4_3_36_60_conv hanoi_4_3_36_81_conv \
         ::: problem-instances/*/latplan.puzzles.hanoi/* \
         ::: Astar



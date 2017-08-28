#!/bin/bash

ulimit -v 16000000000

trap exit SIGINT

ncpu=$(cat /proc/cpuinfo | grep -c processor)
jobs=$((ncpu-1))

parallel -j $jobs --eta --timeout 90 --joblog latplan.puzzles.puzzle_mnist.log  \
         "./trivial-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.log" \
         ::: puzzle_mnist_3_3_36_20000_conv \
         ::: problem-instances/latplan.puzzles.puzzle_mnist/* \
         ::: Astar AstarRec

parallel -j $jobs --eta --timeout 90 --joblog latplan.puzzles.puzzle_mandrill.log   \
         "./trivial-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.log" \
         ::: puzzle_mandrill_3_3_36_20000_conv \
         ::: problem-instances/latplan.puzzles.puzzle_mandrill/* \
         ::: Astar AstarRec


parallel -j $jobs --eta --timeout 90 --joblog latplan.puzzles.puzzle_spider.log   \
         "./trivial-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.log" \
         ::: puzzle_spider_3_3_36_20000_conv \
         ::: problem-instances/latplan.puzzles.puzzle_spider/* \
         ::: Astar AstarRec


parallel -j $jobs --eta --timeout 90 --joblog latplan.puzzles.hanoi.log   \
         "./trivial-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.log" \
         ::: hanoi_7_4_36_10000_conv \
         ::: problem-instances/latplan.puzzles.hanoi/* \
         ::: Astar AstarRec


parallel -j $jobs --eta --timeout 90 --joblog latplan.puzzles.lightsout_digital.log   \
         "./trivial-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.log" \
         ::: lightsout_digital_4_36_20000_conv \
         ::: problem-instances/latplan.puzzles.lightsout_digital/* \
         ::: Astar AstarRec


parallel -j $jobs --eta --timeout 90 --joblog latplan.puzzles.lightsout_twisted.log   \
         "./trivial-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.log" \
         ::: lightsout_twisted_4_36_20000_conv \
         ::: problem-instances/latplan.puzzles.lightsout_twisted/* \
         ::: Astar AstarRec



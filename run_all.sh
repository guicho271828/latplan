#!/bin/bash

trap exit SIGINT

parallel --eta --timeout 90 --joblog latplan.puzzles.puzzle_mnist.log  \
         "./trivial-planner.py samples/{1} {2} Astar > {2}/{1}_Astar.log" \
         ::: puzzle_mnist_3_3_36_20000_conv \
         ::: problem-instances/latplan.puzzles.puzzle_mnist/*

parallel --eta --timeout 90 --joblog latplan.puzzles.puzzle_mandrill.log   \
         "./trivial-planner.py samples/{1} {2} Astar > {2}/{1}_Astar.log" \
         ::: puzzle_mandrill_3_3_36_20000_conv \
         ::: problem-instances/latplan.puzzles.puzzle_mandrill/*

parallel --eta --timeout 90 --joblog latplan.puzzles.puzzle_spider.log   \
         "./trivial-planner.py samples/{1} {2} Astar > {2}/{1}_Astar.log" \
         ::: puzzle_spider_3_3_36_20000_conv \
         ::: problem-instances/latplan.puzzles.puzzle_spider/*

parallel --eta --timeout 90 --joblog latplan.puzzles.hanoi.log   \
         "./trivial-planner.py samples/{1} {2} Astar > {2}/{1}_Astar.log" \
         ::: hanoi_7_4_36_10000_conv \
         ::: problem-instances/latplan.puzzles.hanoi/*

parallel --eta --timeout 90 --joblog latplan.puzzles.lightsout_digital.log   \
         "./trivial-planner.py samples/{1} {2} Astar > {2}/{1}_Astar.log" \
         ::: lightsout_digital_4_36_20000_conv \
         ::: problem-instances/latplan.puzzles.lightsout_digital/*

parallel --eta --timeout 90 --joblog latplan.puzzles.lightsout_twisted.log   \
         "./trivial-planner.py samples/{1} {2} Astar > {2}/{1}_Astar.log" \
         ::: lightsout_twisted_4_36_20000_conv \
         ::: problem-instances/latplan.puzzles.lightsout_twisted/*


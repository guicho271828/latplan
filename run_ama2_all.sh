#!/bin/bash

ulimit -v 16000000000

trap exit SIGINT

parallel --eta --timeout 90 --joblog problem-instances/latplan.puzzles.puzzle_mnist.csv  \
         "./trivial-planner.py {1} {2} {3} > {2}/{1/}_{3}.log" \
         ::: samples/puzzle_mnist* \
         ::: problem-instances/*/latplan.puzzles.puzzle_mnist/* \
         ::: Astar

parallel --eta --timeout 90 --joblog problem-instances/latplan.puzzles.puzzle_mandrill.csv   \
         "./trivial-planner.py {1} {2} {3} > {2}/{1/}_{3}.log" \
         ::: samples/puzzle_mandrill* \
         ::: problem-instances/*/latplan.puzzles.puzzle_mandrill/* \
         ::: Astar


parallel --eta --timeout 90 --joblog problem-instances/latplan.puzzles.puzzle_spider.csv   \
         "./trivial-planner.py {1} {2} {3} > {2}/{1/}_{3}.log" \
         ::: samples/puzzle_spider* \
         ::: problem-instances/*/latplan.puzzles.puzzle_spider/* \
         ::: Astar


parallel --eta --timeout 90 --joblog problem-instances/latplan.puzzles.lightsout_digital.csv   \
         "./trivial-planner.py {1} {2} {3} > {2}/{1/}_{3}.log" \
         ::: samples/lightsout_digital* \
         ::: problem-instances/*/latplan.puzzles.lightsout_digital/* \
         ::: Astar


parallel --eta --timeout 90 --joblog problem-instances/latplan.puzzles.lightsout_twisted.csv   \
         "./trivial-planner.py {1} {2} {3} > {2}/{1/}_{3}.log" \
         ::: samples/lightsout_twisted* \
         ::: problem-instances/*/latplan.puzzles.lightsout_twisted/* \
         ::: Astar


# parallel --eta --timeout 90 --joblog problem-instances/latplan.puzzles.hanoi.csv   \
#          "./trivial-planner.py {1} {2} {3} > {2}/{1/}_{3}.log" \
#          ::: samples/hanoi* \
#          ::: problem-instances/*/latplan.puzzles.hanoi/* \
#          ::: Astar



#!/bin/bash

ulimit -v 16000000000

trap exit SIGINT

common="jbsub -mem 8g -queue x86_1h -proj $(date -Iminutes)"

dir=$(dirname $(dirname $(readlink -ef $0)))
export PYTHONPATH=$dir:$PYTHONPATH
export PYTHONUNBUFFERED=1

parallel   "[ -f {2}/{1/}_{3}_path_0.valid ] || $common './trivial-planner.py {1} {2} {3}  > {2}/{1/}_{3}.log'" \
         ::: $(ls -d samples/puzzle*mnist* | grep -v 1000 | grep -v convn ) \
         ::: noise-0.6-0.12-ama2/*/latplan.puzzles.puzzle_mnist/* \
         ::: Astar

parallel   "[ -f {2}/{1/}_{3}_path_0.valid ] || $common './trivial-planner.py {1} {2} {3}  > {2}/{1/}_{3}.log'" \
         ::: $(ls -d samples/puzzle*mandrill* | grep -v 1000 | grep -v convn ) \
         ::: noise-0.6-0.12-ama2/*/latplan.puzzles.puzzle_mandrill/* \
         ::: Astar


parallel   "[ -f {2}/{1/}_{3}_path_0.valid ] || $common './trivial-planner.py {1} {2} {3}  > {2}/{1/}_{3}.log'" \
         ::: $(ls -d samples/puzzle*spider* | grep -v 1000 | grep -v convn ) \
         ::: noise-0.6-0.12-ama2/*/latplan.puzzles.puzzle_spider/* \
         ::: Astar


parallel   "[ -f {2}/{1/}_{3}_path_0.valid ] || $common './trivial-planner.py {1} {2} {3}  > {2}/{1/}_{3}.log'" \
         ::: $(ls -d samples/lightsout*digital* | grep -v 1000 | grep -v convn ) \
         ::: noise-0.6-0.12-ama2/*/latplan.puzzles.lightsout_digital/* \
         ::: Astar


parallel   "[ -f {2}/{1/}_{3}_path_0.valid ] || $common './trivial-planner.py {1} {2} {3}  > {2}/{1/}_{3}.log'" \
         ::: $(ls -d samples/lightsout*twisted* | grep -v 1000 | grep -v convn ) \
         ::: noise-0.6-0.12-ama2/*/latplan.puzzles.lightsout_twisted/* \
         ::: Astar


# parallel   "[ -f {2}/{1/}_{3}_path_0.valid ] || $common './trivial-planner.py {1} {2} {3}  > {2}/{1/}_{3}.log'" \
#          ::: samples/hanoi* \
#          ::: noise-0.6-0.12-ama2/*/latplan.puzzles.hanoi/* \
#          ::: Astar



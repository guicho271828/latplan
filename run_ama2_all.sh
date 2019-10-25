#!/bin/bash

ulimit -v 16000000000

trap exit SIGINT

proj=$(date +%Y%m%d%H%M)
common="jbsub -mem 8g -queue x86_1h -proj $proj"

dir=$(dirname $(dirname $(readlink -ef $0)))
export PYTHONPATH=$dir:$PYTHONPATH
export PYTHONUNBUFFERED=1

parallel   "[ -f {2}/{1/}_{3}_path_0.valid ] || $common './ama2-planner.py {1} {2} {3}  > {2}/{1/}_{3}.log'" \
         ::: $(ls -d samples/puzzle*mnist* ) \
         ::: noise-0.6-0.12-ama2/*/latplan.puzzles.puzzle_mnist/* \
         ::: Astar

parallel   "[ -f {2}/{1/}_{3}_path_0.valid ] || $common './ama2-planner.py {1} {2} {3}  > {2}/{1/}_{3}.log'" \
         ::: $(ls -d samples/puzzle*mandrill* ) \
         ::: noise-0.6-0.12-ama2/*/latplan.puzzles.puzzle_mandrill/* \
         ::: Astar


parallel   "[ -f {2}/{1/}_{3}_path_0.valid ] || $common './ama2-planner.py {1} {2} {3}  > {2}/{1/}_{3}.log'" \
         ::: $(ls -d samples/puzzle*spider* ) \
         ::: noise-0.6-0.12-ama2/*/latplan.puzzles.puzzle_spider/* \
         ::: Astar


parallel   "[ -f {2}/{1/}_{3}_path_0.valid ] || $common './ama2-planner.py {1} {2} {3}  > {2}/{1/}_{3}.log'" \
         ::: $(ls -d samples/lightsout*digital* ) \
         ::: noise-0.6-0.12-ama2/*/latplan.puzzles.lightsout_digital/* \
         ::: Astar


parallel   "[ -f {2}/{1/}_{3}_path_0.valid ] || $common './ama2-planner.py {1} {2} {3}  > {2}/{1/}_{3}.log'" \
         ::: $(ls -d samples/lightsout*twisted* ) \
         ::: noise-0.6-0.12-ama2/*/latplan.puzzles.lightsout_twisted/* \
         ::: Astar


# parallel   "[ -f {2}/{1/}_{3}_path_0.valid ] || $common './ama2-planner.py {1} {2} {3}  > {2}/{1/}_{3}.log'" \
#          ::: samples/hanoi* \
#          ::: noise-0.6-0.12-ama2/*/latplan.puzzles.hanoi/* \
#          ::: Astar



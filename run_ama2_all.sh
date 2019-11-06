#!/bin/bash -x

ulimit -v 16000000000

trap exit SIGINT

probdir=problem-instances

#### foolproof check

# ensuring if the system is built correctly
(
    make -C lisp
    git submodule update --init --recursive
    cd downward
    ./build.py -j $(cat /proc/cpuinfo | grep -c processor) release64
)

chmod -R +w $probdir            # in the weird case this happens

#### job submission

proj=$(date +%Y%m%d%H%M)ama2
command="jbsub -mem 8g -queue x86_1h -proj $proj task"
export PYTHONPATH=$(dirname $(dirname $(readlink -ef $0))):$PYTHONPATH
export PYTHONUNBUFFERED=1
export SHELL=/bin/bash
task (){
    ./ama2-planner.py \
        $@ \
        1> $2/ama2_$(basename $1)_$3.log \
        2> $2/ama2_$(basename $1)_$3.err
}
export -f task

parallel $command \
         ::: samples/puzzle*mnist* \
         ::: $probdir/*/latplan.puzzles.puzzle_mnist/* \
         ::: Astar

parallel $command \
         ::: samples/puzzle*mandrill* \
         ::: $probdir/*/latplan.puzzles.puzzle_mandrill/* \
         ::: Astar


parallel $command \
         ::: samples/puzzle*spider*  \
         ::: $probdir/*/latplan.puzzles.puzzle_spider/* \
         ::: Astar


parallel $command \
         ::: samples/lightsout*digital*  \
         ::: $probdir/*/latplan.puzzles.lightsout_digital/* \
         ::: Astar


parallel $command \
         ::: samples/lightsout*twisted*  \
         ::: $probdir/*/latplan.puzzles.lightsout_twisted/* \
         ::: Astar

# parallel $command \
#          ::: samples/hanoi* \
#          ::: $probdir/*/latplan.puzzles.hanoi/* \
#          ::: Astar



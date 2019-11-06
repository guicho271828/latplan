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

key=$1
if [ -z $key ]
then
    echo "usage: $0 domainfilename [mem]" >&2
    echo "mem: e.g. 64g" >&2
    exit 1
fi
mem=${2:-64g}

proj=$(date +%Y%m%d%H%M)ama3
command="jbsub -mem $mem -queue x86_1h -proj $proj task"
export PYTHONPATH=$(dirname $(dirname $(readlink -ef $0))):$PYTHONPATH
export PYTHONUNBUFFERED=1
task (){
    ./ama3-planner.py \
        $@ \
        1> $2/ama3_$(basename $(dirname $1))_$(basename $1 .pddl)_$3.log \
        2> $2/ama3_$(basename $(dirname $1))_$(basename $1 .pddl)_$3.err
}
export -f task

parallel $command \
         ::: samples/puzzle*mnist*/${key}.pddl \
         ::: $probdir/*/latplan.puzzles.puzzle_mnist/* \
         ::: blind 

parallel $command \
         ::: samples/puzzle*mandrill*/${key}.pddl \
         ::: $probdir/*/latplan.puzzles.puzzle_mandrill/* \
         ::: blind 

parallel $command \
         ::: samples/puzzle*spider*/${key}.pddl \
         ::: $probdir/*/latplan.puzzles.puzzle_spider/* \
         ::: blind 

parallel $command \
         ::: samples/lightsout*digital*/${key}.pddl \
         ::: $probdir/*/latplan.puzzles.lightsout_digital/* \
         ::: blind 

parallel $command \
         ::: samples/lightsout*twisted*/${key}.pddl \
         ::: $probdir/*/latplan.puzzles.lightsout_twisted/* \
         ::: blind 

# parallel -j 1 --no-notice \
#          "jbsub $common './ama3-planner.py {1} {2} {3} > {2}/{1/}_{3}.ama3.log 2> {2}/{1/}_{3}.ama3.err'" \
#          ::: samples/hanoi* \
#          ::: $probdir/*/latplan.puzzles.hanoi/* \
#          ::: blind  \
         # ::: remlic-1-1-0 remlic-2-2-0 remlic-4-4-0  actionlearner rf-2-others1b-t rf-5-others1b-t rf-10-others1b-t



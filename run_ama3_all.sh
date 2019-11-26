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
mem=${2:-256g}

proj=$(date +%Y%m%d%H%M)ama3
command="jbsub -mem $mem -queue x86_1h -proj $proj task"
export PYTHONPATH=$(dirname $(dirname $(readlink -ef $0))):$PYTHONPATH
export PYTHONUNBUFFERED=1
task (){
    out=$(echo ${1%%.pddl} | sed 's@/@_@g')
    ./ama3-planner.py \
        $@ \
        1> $2/ama3_${out}_$3.log \
        2> $2/ama3_${out}_$3.err
}
export -f task

parallel $command \
         ::: samples/puzzle*mnist*/${key}.pddl \
         ::: $probdir/*/latplan.puzzles.puzzle_mnist/* \
         ::: ff hmax 

parallel $command \
         ::: samples/puzzle*mandrill*/${key}.pddl \
         ::: $probdir/*/latplan.puzzles.puzzle_mandrill/* \
         ::: ff hmax 

parallel $command \
         ::: samples/puzzle*spider*/${key}.pddl \
         ::: $probdir/*/latplan.puzzles.puzzle_spider/* \
         ::: ff hmax 

parallel $command \
         ::: samples/lightsout*digital*/${key}.pddl \
         ::: $probdir/*/latplan.puzzles.lightsout_digital/* \
         ::: ff hmax 

parallel $command \
         ::: samples/lightsout*twisted*/${key}.pddl \
         ::: $probdir/*/latplan.puzzles.lightsout_twisted/* \
         ::: ff hmax 

# parallel -j 1 --no-notice \
#          "jbsub $common './ama3-planner.py {1} {2} {3} > {2}/{1/}_{3}.ama3.log 2> {2}/{1/}_{3}.ama3.err'" \
#          ::: samples/hanoi* \
#          ::: $probdir/*/latplan.puzzles.hanoi/* \
#          ::: ff hmax  \
         # ::: remlic-1-1-0 remlic-2-2-0 remlic-4-4-0  actionlearner rf-2-others1b-t rf-5-others1b-t rf-10-others1b-t



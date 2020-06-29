#!/bin/bash -x

ulimit -v 16000000000

trap exit SIGINT

probdir=problem-instances-ama2

#### foolproof check

# ensuring if the system is built correctly

chmod -R +w $probdir            # in the weird case this happens

#### job submission

proj=$(date +%Y%m%d%H%M)ama2
command="jbsub -mem 8g -queue x86_1h -proj $proj task"
export PYTHONUNBUFFERED=1
export SHELL=/bin/bash
task (){
    base=$2/ama2_$(basename $1)_$3_$5
    outfile=$base.log
    errfile=$base.err
    trap "cat $outfile; cat $errfile >&2" RETURN
    planner-scripts/timeout/timeout -t 900 ./ama2-planner.py \
        $@ \
        1> $outfile \
        2> $errfile
}
export -f task

parallel $command \
         ::: samples-for-aae/puzzle*mnist* \
         ::: $probdir/*/latplan.puzzles.puzzle_mnist/* \
         ::: Astar ::: False ::: blind goalcount ::: _ActionAE_None _CubeActionAE_None

parallel $command \
         ::: samples-for-aae/puzzle*mandrill* \
         ::: $probdir/*/latplan.puzzles.puzzle_mandrill/* \
         ::: Astar ::: False ::: blind goalcount ::: _ActionAE_None _CubeActionAE_None


parallel $command \
         ::: samples-for-aae/puzzle*spider*  \
         ::: $probdir/*/latplan.puzzles.puzzle_spider/* \
         ::: Astar ::: False ::: blind goalcount ::: _ActionAE_None _CubeActionAE_None


parallel $command \
         ::: samples-for-aae/lightsout*digital*  \
         ::: $probdir/*/latplan.puzzles.lightsout_digital/* \
         ::: Astar ::: False ::: blind goalcount ::: _ActionAE_None _CubeActionAE_None


parallel $command \
         ::: samples-for-aae/lightsout*twisted*  \
         ::: $probdir/*/latplan.puzzles.lightsout_twisted/* \
         ::: Astar ::: False ::: blind goalcount ::: _ActionAE_None _CubeActionAE_None

# parallel $command \
#          ::: samples-for-aae/hanoi* \
#          ::: $probdir/*/latplan.puzzles.hanoi/* \
#          ::: Astar ::: False ::: blind goalcount ::: _ActionAE_None _CubeActionAE_None



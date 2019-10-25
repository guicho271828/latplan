#!/bin/bash -x

set -e

trap exit SIGINT

export SHELL=/bin/bash
ulimit -v 16000000000

dir=$(dirname $(dirname $(readlink -ef $0)))

proj1=$(date +%Y%m%d%H%M)aae
common="jbsub -mem 64g -cores 1+1 -queue x86_1h -proj $proj1"

parallel $common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 {} \
         ::: ./action_autoencoder.py \
         ::: samples/* \
         ::: learn_test_dump

proj2=$(date +%Y%m%d%H%M)sd
common="jbsub -mem 64g -cores 1+1 -queue x86_6h -proj $proj2"
parallel $common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 {} \
         ::: ./state_discriminator3.py \
         ::: samples/* \
         ::: learn_test_dump \
         ::: direct

watch-proj $proj1
watch-proj $proj2

proj=$(date +%Y%m%d%H%M)ad
common="jbsub -mem 64g -cores 1+1 -queue x86_1h -proj $proj"
parallel $common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 {} \
         ::: ./action_discriminator.py \
         ::: samples/* \
         ::: learn_test_dump


proj=$(date +%Y%m%d%H%M)
common="jbsub -mem 16g -cores 1 -queue x86_1h -proj $proj"

make -C lisp -j 1

actionlearner (){
    lisp/domain-actionlearner.bin $1/actions.csv > $1/actionlearner.pddl 2>$1/actionlearner.err
}

export -f actionlearner

parallel $common actionlearner ::: samples/*/


common="jbsub -mem 16g -cores 8 -queue x86_6h -proj $proj"

export ntrees="80 40 20 10 5 2 1"
export depths="100 50 25 12 7 4"

dsama-effect (){
    dir=$1
    [ -e $dir/action+ids.csv ]       || (tr "[:blank:]" "\t" < $dir/actions.csv | paste - $dir/action_ids.csv > $dir/action+ids.csv)
    [ -e $dir/action+ids.fasl ]      || (dsama dump-tsv $dir/action+ids.csv $dir/action+ids.fasl )

    echo $dir
    echo $(readlink -ef $dir)
    cd $dir
    shift
    actions=$(jq .parameters.M _aae/aux.json)
    parallel -j 16 --verbose "dsama train-model action+ids.fasl \"(:effect {} $@)\"" \
             ::: $(seq $actions) \
             ::: binary-random-forest-classifier \
             ::: :n-tree \
             ::: $ntrees \
             ::: :bagging-ratio \
             ::: 0.34 \
             ::: :max-depth \
             ::: $depths
    true
}

export -f dsama-effect

parallel $common dsama-effect ::: samples/*/


common="jbsub -mem 16g -cores 8 -queue x86_6h -proj $proj"

dsama-precondition (){
    dir=$1
    [ -e $dir/action+ids.csv ]       || (tr "[:blank:]" "\t" < $dir/actions.csv | paste - $dir/action_ids.csv > $dir/action+ids.csv)
    [ -e $dir/action+ids.fasl ]      || (dsama dump-tsv $dir/action+ids.csv $dir/action+ids.fasl )

    echo $dir
    echo $(readlink -ef $dir)
    cd $dir
    shift
    actions=$(jq .parameters.M _aae/aux.json)
    parallel -j 16 --verbose "dsama train-model action+ids.fasl \"(:precondition {} $@)\"" \
             ::: $(seq $actions) \
             ::: binary-random-forest-classifier pu-random-forest-classifier \
             ::: :n-tree \
             ::: $ntrees \
             ::: :bagging-ratio \
             ::: 0.34 \
             ::: :max-depth \
             ::: $depths
    true
}

export -f dsama-precondition

parallel $common dsama-precondition ::: samples/*/



watch-proj $proj

proj=$(date +%Y%m%d%H%M)accuracy
common="jbsub -mem 16g -cores 8 -queue x86_1h -proj $proj"

rf-accuracy (){
    ./rf-accuracy.ros --pre $2-RANDOM-FOREST-CLASSIFIER-N-TREE-$4-BAGGING-RATIO-0.34-MAX-DEPTH-$5.json \
                      --eff $3-RANDOM-FOREST-CLASSIFIER-N-TREE-$4-BAGGING-RATIO-0.34-MAX-DEPTH-$5.json \
                      $1 | tee $1$2-$3-$4-$5.csv
}

export -f rf-accuracy

parallel $common rf-accuracy  \
         ::: samples/*/ \
         ::: PU \
         ::: BINARY \
         ::: $ntrees \
         ::: $depths

proj=$(date +%Y%m%d%H%M)pddl
common="jbsub -mem 16g -cores 1 -queue x86_1h -proj $proj"

to-pddl (){
    models=$(mktemp)
    trap "rm $models" RETURN
    cd $1
    # ls -v is necessary because bash will reorder them in a wrong manner e.g. 1, 10, 11, 12, ... 
    parallel --keep-order "echo {1}/best.model \$(ls -v {2}/*/best.model)" \
             ::: $(ls -dv PRECONDITION/*) \
             :::+ $(ls -dv EFFECT/*) \
             > $models
    dsama to-pddl $models dsama.pddl
}

export -f to-pddl

to-pddl-custom (){
    models=$(mktemp)
    trap "rm $models" RETURN
    cd $1
    pre=$2
    eff=$3
    ntree=$4
    depth=$5
    # ls -v is necessary because bash will reorder them in a wrong manner e.g. 1, 10, 11, 12, ... 
    parallel --keep-order "echo {1}/${pre}-RANDOM-FOREST-CLASSIFIER-N-TREE-${ntree}-BAGGING-RATIO-0.34-MAX-DEPTH-${depth}.model \$(ls -v {2}/*/${eff}-RANDOM-FOREST-CLASSIFIER-N-TREE-${ntree}-BAGGING-RATIO-0.34-MAX-DEPTH-${depth}.model)" \
             ::: $(ls -dv PRECONDITION/*) \
             :::+ $(ls -dv EFFECT/*) \
             > $models
    dsama to-pddl $models dsama-custom-$pre-$eff-$ntree-$depth.pddl
}

export -f to-pddl-custom

# parallel $common ::: to-pddl ::: samples/*/
parallel $common ::: to-pddl-custom ::: samples/*/ ::: PU ::: BINARY ::: $ntrees ::: $depths


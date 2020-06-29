#!/bin/bash -x

make -j 1 -C lisp

set -e

trap exit SIGINT

ulimit -v 16000000000

export PYTHONUNBUFFERED=1




train-ad-unused (){
    proj=$(date +%Y%m%d%H%M)sd
    common="jbsub -mem 64g -cores 1+1 -queue x86_6h -proj $proj"
    parallel $common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 {} \
             ::: ./state_discriminator3.py \
             ::: $base/* \
             ::: learn_test_dump \
             ::: direct

    watch-proj $proj # wait

    proj=$(date +%Y%m%d%H%M)ad
    common="jbsub -mem 64g -cores 1+1 -queue x86_1h -proj $proj"
    parallel $common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 {} \
             ::: ./action_discriminator.py \
             ::: $base/* \
             ::: learn_test_dump

    proj=$(date +%Y%m%d%H%M)ad2
    common="jbsub -mem 64g -cores 1+1 -queue x86_1h -proj $proj"
    parallel $common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 {} \
             ::: ./action_discriminator.py \
             ::: $base/* \
             ::: learn_test_dump \
             ::: prepare_aae_PU4
}

actionlearner (){
    lisp/domain-actionlearner.bin $1/actions.csv > $1/actionlearner.pddl 2>$1/actionlearner.err
    lisp/domain-actionlearner.bin $1/actions_aae.csv > $1/actionlearner_aae.pddl 2>$1/actionlearner_aae.err
}

export -f actionlearner

actionlearner2 (){
    lisp/domain-actionlearner2.bin $@ > $1/actionlearner2-$2-$3.pddl 2>$1/actionlearner2-$2-$3.err
}

export -f actionlearner2

actionlearner3 (){
    lisp/domain-actionlearner3.bin $@ > $1/actionlearner3-$2-$3-$4.pddl 2>$1/actionlearner3-$2-$3-$4.err
}

export -f actionlearner3

actionlearner4 (){
    lisp/domain-actionlearner4.bin $@ > $1/actionlearner4-$2-$3-$4-$5.pddl 2>$1/actionlearner4-$2-$3-$4-$5.err
}

export -f actionlearner4

dsama-dump (){
    [ -f ${1%%csv}fasl ] || dsama dump-tsv $1 ${1%%csv}fasl
}

export -f dsama-dump

export ntrees="80 20 5 1"
export depths="100 25 12 4"
export bagging="0.66"

dsama-precondition (){
    dir=$1
    echo $dir
    echo $(readlink -ef $dir)
    cd $dir
    shift
    parallel -j 16 --verbose "[ -f PRECONDITION/{1}/PU-RANDOM-FOREST-CLASSIFIER-N-TREE-{4}-BAGGING-RATIO-{6}-MAX-DEPTH-{8}.model ] || dsama train-precondition actions+ids.fasl fake_actions+ids.fasl valid_actions+ids.fasl invalid_actions+ids.fasl {}" \
             ::: $(cat available_actions.csv) \
             ::: pu-random-forest-classifier \
             ::: :n-tree \
             ::: $ntrees \
             ::: :bagging-ratio \
             ::: ${bagging} \
             ::: :max-depth \
             ::: $depths
    true
}

export -f dsama-precondition

dsama-precondition-with-successors (){
    dir=$1
    echo $dir
    echo $(readlink -ef $dir)
    cd $dir
    shift
    parallel -j 16 --verbose "[ -f PRECONDITION-WITH-SUCCESSORS/{1}/PU-RANDOM-FOREST-CLASSIFIER-N-TREE-{4}-BAGGING-RATIO-{6}-MAX-DEPTH-{8}.model ] || dsama train-precondition-with-successors actions+ids.fasl fake_actions+ids.fasl valid_actions+ids.fasl invalid_actions+ids.fasl {}" \
             ::: $(cat available_actions.csv) \
             ::: pu-random-forest-classifier \
             ::: :n-tree \
             ::: $ntrees \
             ::: :bagging-ratio \
             ::: ${bagging} \
             ::: :max-depth \
             ::: $depths
    true
}

export -f dsama-precondition-with-successors

rf-accuracy (){
    mkdir -p $1accuracy
    ./rf-accuracy.ros --pre $2-RANDOM-FOREST-CLASSIFIER-N-TREE-$4-BAGGING-RATIO-${bagging}-MAX-DEPTH-$5.json \
                      --eff $3-RANDOM-FOREST-CLASSIFIER-N-TREE-$4-BAGGING-RATIO-${bagging}-MAX-DEPTH-$5.json \
                      $1 | tee $1accuracy/$2-$3-$4-$5.csv
}

export -f rf-accuracy

rf-accuracy-with-successors (){
    mkdir -p $1accuracy-with-successors
    ./rf-accuracy.ros --pre $2-RANDOM-FOREST-CLASSIFIER-N-TREE-$4-BAGGING-RATIO-${bagging}-MAX-DEPTH-$5.json \
                      --eff $3-RANDOM-FOREST-CLASSIFIER-N-TREE-$4-BAGGING-RATIO-${bagging}-MAX-DEPTH-$5.json \
                      --predir PRECONDITION-WITH-SUCCESSORS/ \
                      $1 | tee $1accuracy-with-successors/$2-$3-$4-$5.csv
}

export -f rf-accuracy-with-successors

to-pddl (){
    models=$(mktemp)
    trap "rm $models" RETURN
    cd $1
    pre=$2
    eff=$3
    ntree=$4
    depth=$5
    mkdir -p PDDL
    if ! [ -f PDDL/dsama-$pre-$eff-$ntree-$depth.pddl ]
    then
        # ls -v is necessary because bash will reorder them in a wrong manner e.g. 1, 10, 11, 12, ... 
        parallel --keep-order "echo {1}/${pre}-RANDOM-FOREST-CLASSIFIER-N-TREE-${ntree}-BAGGING-RATIO-${bagging}-MAX-DEPTH-${depth}.model \$(ls -v {2}/*/${eff}-RANDOM-FOREST-CLASSIFIER-N-TREE-${ntree}-BAGGING-RATIO-${bagging}-MAX-DEPTH-${depth}.model)" \
                 ::: $(ls -dv PRECONDITION/*) \
                 :::+ $(ls -dv EFFECT/*) \
                 > $models
        dsama to-pddl $models PDDL/dsama-$pre-$eff-$ntree-$depth.pddl
    fi
}

export -f to-pddl

to-pddl-with-successors (){
    models=$(mktemp)
    trap "rm $models" RETURN
    cd $1
    pre=$2
    eff=$3
    ntree=$4
    depth=$5
    mkdir -p PDDL-WITH-SUCCESSORS
    if ! [ -f PDDL-WITH-SUCCESSORS/dsama-$pre-$eff-$ntree-$depth.pddl ]
    then
        # ls -v is necessary because bash will reorder them in a wrong manner e.g. 1, 10, 11, 12, ... 
        parallel --keep-order "echo {1}/${pre}-RANDOM-FOREST-CLASSIFIER-N-TREE-${ntree}-BAGGING-RATIO-${bagging}-MAX-DEPTH-${depth}.model \$(ls -v {2}/*/${eff}-RANDOM-FOREST-CLASSIFIER-N-TREE-${ntree}-BAGGING-RATIO-${bagging}-MAX-DEPTH-${depth}.model)" \
                 ::: $(ls -dv PRECONDITION-WITH-SUCCESSORS/*) \
                 :::+ $(ls -dv EFFECT/*) \
                 > $models
        dsama to-pddl $models PDDL-WITH-SUCCESSORS/dsama-$pre-$eff-$ntree-$depth.pddl
    fi
}

export -f to-pddl-with-successors

to-pddl2 (){
    models=$(mktemp)
    trap "rm $models" RETURN
    cd $1
    pre=$2
    eff=$3
    ntree=$4
    depth=$5
    mkdir -p PDDL2
    if ! [ -f PDDL2/dsama-$pre-$eff-$ntree-$depth.pddl ]
    then
        # ls -v is necessary because bash will reorder them in a wrong manner e.g. 1, 10, 11, 12, ... 
        parallel --keep-order "echo {1}/${pre}-RANDOM-FOREST-CLASSIFIER-N-TREE-${ntree}-BAGGING-RATIO-${bagging}-MAX-DEPTH-${depth}.model" \
                 ::: $(ls -dv PRECONDITION/*) \
                 > $models
        dsama to-pddl2 $models action_add$eff.csv action_del$eff.csv PDDL2/dsama-$pre-$eff-$ntree-$depth.pddl
    fi
}

export -f to-pddl2

to-pddl2-with-successors (){
    models=$(mktemp)
    trap "rm $models" RETURN
    cd $1
    pre=$2
    eff=$3
    ntree=$4
    depth=$5
    mkdir -p PDDL2-WITH-SUCCESSORS
    if ! [ -f PDDL2-WITH-SUCCESSORS/dsama-$pre-$eff-$ntree-$depth.pddl ]
    then
        # ls -v is necessary because bash will reorder them in a wrong manner e.g. 1, 10, 11, 12, ... 
        parallel --keep-order "echo {1}/${pre}-RANDOM-FOREST-CLASSIFIER-N-TREE-${ntree}-BAGGING-RATIO-${bagging}-MAX-DEPTH-${depth}.model" \
                 ::: $(ls -dv PRECONDITION-WITH-SUCCESSORS/*) \
                 > $models
        dsama to-pddl2 $models action_add$eff.csv action_del$eff.csv PDDL2-WITH-SUCCESSORS/dsama-$pre-$eff-$ntree-$depth.pddl
    fi
}

export -f to-pddl2-with-successors

to-pddl2-nopre (){
    cd $1
    eff=$2
    mkdir -p PDDL2-NOPRE
    if ! [ -f PDDL2-NOPRE/dsama-$eff.pddl ]
    then
        dsama to-pddl2-nopre action_add$eff.csv action_del$eff.csv PDDL2-NOPRE/dsama-$eff.pddl
    fi
}

export -f to-pddl2-nopre



base=samples-planning
suffix="*"

proj=$(date +%Y%m%d%H%M)al4-$base-$suffix
common="jbsub -mem 32g -cores 16 -queue x86_6h -proj $proj"
# parallel $common actionlearner4 ::: $base/*$suffix/ ::: 3 ::: actions_both+ids.csv ::: 0 ::: 1.00

base=samples-16
suffix="*"

proj=$(date +%Y%m%d%H%M)al4-$base-$suffix
common="jbsub -mem 32g -cores 16 -queue x86_6h -proj $proj"
# parallel $common actionlearner4 ::: $base/*$suffix/ ::: 3 ::: actions_both+ids.csv ::: 0 ::: 1.00


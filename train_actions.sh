#!/bin/bash -x

set -e

trap exit SIGINT

export SHELL=/bin/bash
ulimit -v 16000000000
make -C lisp -j 1


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

watch-proj $proj1 $proj2 # wait

proj=$(date +%Y%m%d%H%M)ad
common="jbsub -mem 64g -cores 1+1 -queue x86_1h -proj $proj"
parallel $common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 {} \
         ::: ./action_discriminator.py \
         ::: samples/* \
         ::: learn_test_dump

proj=$(date +%Y%m%d%H%M)ad2
common="jbsub -mem 64g -cores 1+1 -queue x86_1h -proj $proj"
parallel $common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 {} \
         ::: ./action_discriminator.py \
         ::: samples/* \
         ::: learn_test_dump \
         ::: prepare_aae_PU4 \
         ::: _ad2/

actionlearner (){
    lisp/domain-actionlearner.bin $1/actions.csv > $1/actionlearner.pddl 2>$1/actionlearner.err
}

export -f actionlearner

dsama-dump (){
    dsama dump-tsv $1 ${1%%csv}fasl
}

export -f dsama-dump

export ntrees="80 40 20 10 5 2 1"
export depths="100 50 25 18 12 7 4"
export bagging="0.66"

dsama-effect (){
    dir=$1
    echo $dir
    echo $(readlink -ef $dir)
    cd $dir
    shift
    parallel -j 16 --verbose dsama train-effect actions+ids.fasl \
             ::: $(cat available_actions.csv) \
             ::: binary-random-forest-classifier \
             ::: :n-tree \
             ::: $ntrees \
             ::: :bagging-ratio \
             ::: ${bagging} \
             ::: :max-depth \
             ::: $depths
    true
}

export -f dsama-effect

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

proj2=$(date +%Y%m%d%H%M)pre
common="jbsub -mem 32g -cores 16 -queue x86_6h -proj $proj2"
parallel $common dsama-precondition                 ::: samples/*/
parallel $common dsama-precondition-with-successors ::: samples/*/

watch-proj $proj1 $proj2 # wait

proj1=$(date +%Y%m%d%H%M)accuracy
common="jbsub -mem 32g -cores 8 -queue x86_1h -proj $proj1"
parallel $common rf-accuracy                 ::: samples/*/ ::: PU ::: BINARY ::: $ntrees ::: $depths
parallel $common rf-accuracy-with-successors ::: samples/*/ ::: PU ::: BINARY ::: $ntrees ::: $depths

proj2=$(date +%Y%m%d%H%M)pddl
common="jbsub -mem 32g -cores 1 -queue x86_1h -proj $proj2"
parallel $common ::: to-pddl                 ::: samples/*/ ::: PU ::: BINARY ::: $ntrees ::: $depths
parallel $common ::: to-pddl-with-successors ::: samples/*/ ::: PU ::: BINARY ::: $ntrees ::: $depths

watch-proj $proj1 $proj2
./table1.sh > samples/table1
./table2.sh > samples/table2
./table3.sh > samples/table3


#!/bin/bash

set -e

trap exit SIGINT

ulimit -v 16000000000

export PYTHONUNBUFFERED=1
# sokoban problem 2 has the same small screen size as problem 0, and has more than 20000 states unlike problem 0.
# ('sokoban_image-20000-global-global-0-train.npz', array([56, 56,  3]), (3613, 1, 9408)) --- probelm 0 has only 3613 states!
# ('sokoban_image-20000-global-global-2-train.npz', array([56, 56,  3]), (19999, 1, 9408))
export skb_train=sokoban_image-20000-global-global-2-train
export SHELL=/bin/bash
export common

task (){
    script=$1 ; shift
    mode=$1
    # main training experiments. results are used for planning experiments

    $common $script $mode hanoi     4 4           {} $comment ::: 5000 ::: CubeSpaceAE_AMA{3,4}Conv
    $common $script $mode hanoi     3 9           {} $comment ::: 5000 ::: CubeSpaceAE_AMA{3,4}Conv
    $common $script $mode hanoi     4 9           {} $comment ::: 5000 ::: CubeSpaceAE_AMA{3,4}Conv
    $common $script $mode hanoi     5 9           {} $comment ::: 5000 ::: CubeSpaceAE_AMA{3,4}Conv
    $common $script $mode puzzle    mnist    3 3  {} $comment ::: 5000 ::: CubeSpaceAE_AMA{3,4}Conv
    $common $script $mode lightsout digital    5  {} $comment ::: 5000 ::: CubeSpaceAE_AMA{3,4}Conv
    $common $script $mode lightsout twisted    5  {} $comment ::: 5000 ::: CubeSpaceAE_AMA{3,4}Conv
    $common -queue x86_12h $script $mode puzzle    mandrill 4 4  {} $comment ::: 20000 ::: CubeSpaceAE_AMA3Conv
    $common -queue x86_24h $script $mode puzzle    mandrill 4 4  {} $comment ::: 20000 ::: CubeSpaceAE_AMA4Conv
    $common -queue x86_6h  $script $mode sokoban   $skb_train    {} $comment ::: 20000 ::: CubeSpaceAE_AMA3Conv
    $common -queue x86_12h $script $mode sokoban   $skb_train    {} $comment ::: 20000 ::: CubeSpaceAE_AMA4Conv
    $common -queue x86_12h $script $mode blocks    cylinders-4-flat {} $comment ::: 20000 ::: CubeSpaceAE_AMA3Conv
    $common -queue x86_24h $script $mode blocks    cylinders-4-flat {} $comment ::: 20000 ::: CubeSpaceAE_AMA4Conv
}

export -f task

proj=$(date +%Y%m%d%H%M)sae-planning
number=2

################################################################
## Train the network, and run plot, summary, dump for as the job finishes
common="parallel -j 1 --keep-order jbsub -mem 16g -cores 1+1 -queue x86_6h -proj $proj -require 'v100||a100'"

# export comment=kltune$number
# parallel -j 1 --keep-order task ./train_kltune.py learn_summary_plot_dump ::: {1..30}
# export comment=notune$number
# parallel -j 1 --keep-order task ./train_notune.py learn_summary_plot_dump ::: {1..3}
# export comment=nozsae$number
# parallel -j 1 --keep-order task ./train_nozsae.py learn_summary_plot_dump ::: {1..30}


################################################################
## rerun the plot, summary, dump for all hyperparameters
common="parallel -j 1 --keep-order jbsub -mem 32g -cores 1+1 -queue x86_6h -proj $proj -require 'v100||a100'"

# export comment=kltune$number
# parallel -j 1 --keep-order task ./train_kltune.py iterate_summary_plot_dump ::: 1
# export comment=notune$number
# parallel -j 1 --keep-order task ./train_notune.py iterate_summary_plot_dump ::: 1
# export comment=nozsae$number
# parallel -j 1 --keep-order task ./train_nozsae.py iterate_summary_plot_dump ::: 1


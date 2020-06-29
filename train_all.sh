#!/bin/bash -x

set -e

trap exit SIGINT

ulimit -v 16000000000

export PYTHONUNBUFFERED=1

task-planning (){
    mode=$1
    # main training for plannning --- limit=300

    # planning version
    $common ./strips.py $mode puzzle    mnist    3 3 {} "planning" ::: 5000 ::: None ::: None ::: None ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE 
    $common ./strips.py $mode puzzle    spider   3 3 {} "planning" ::: 5000 ::: None ::: None ::: None ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    $common ./strips.py $mode puzzle    mandrill 3 3 {} "planning" ::: 5000 ::: None ::: None ::: None ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    $common ./strips.py $mode lightsout digital    4 {} "planning" ::: 5000 ::: None ::: None ::: None ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    $common ./strips.py $mode lightsout twisted    4 {} "planning" ::: 5000 ::: None ::: None ::: None ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    # $common ./strips.py $mode hanoi     4 8          {} "planning" ::: 5000 ::: None ::: None ::: None ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    $common ./strips.py $mode hanoi     9 3          {} "planning" ::: 5000 ::: None ::: None ::: None ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
}

task-ablation (){
    mode=$1
    # ablation study --- limit=100

    # no direct loss
    # post-experiment note: significant.
    $common ./strips-ablation.py $mode puzzle    mnist    3 3 {} "nodirect" ::: 5000 ::: None ::: None ::: 0.0  ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    $common ./strips-ablation.py $mode puzzle    spider   3 3 {} "nodirect" ::: 5000 ::: None ::: None ::: 0.0  ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    $common ./strips-ablation.py $mode puzzle    mandrill 3 3 {} "nodirect" ::: 5000 ::: None ::: None ::: 0.0  ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    $common ./strips-ablation.py $mode lightsout digital    4 {} "nodirect" ::: 5000 ::: None ::: None ::: 0.0  ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    $common ./strips-ablation.py $mode lightsout twisted    4 {} "nodirect" ::: 5000 ::: None ::: None ::: 0.0  ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    # $common ./strips-ablation.py $mode hanoi     4 8          {} "nodirect" ::: 5000 ::: None ::: None ::: 0.0  ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    $common ./strips-ablation.py $mode hanoi     9 3          {} "nodirect" ::: 5000 ::: None ::: None ::: 0.0  ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE

    # stop gradient
    # post-experiment note: not significant, probably because the decoder can adapt. requires a separate AAE experiment
    # $common ./strips-ablation.py $mode puzzle    mnist    3 3 {} "stopgrad" ::: 5000 ::: None ::: None ::: None ::: True  ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    # $common ./strips-ablation.py $mode puzzle    spider   3 3 {} "stopgrad" ::: 5000 ::: None ::: None ::: None ::: True  ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    # $common ./strips-ablation.py $mode puzzle    mandrill 3 3 {} "stopgrad" ::: 5000 ::: None ::: None ::: None ::: True  ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    # $common ./strips-ablation.py $mode lightsout digital    4 {} "stopgrad" ::: 5000 ::: None ::: None ::: None ::: True  ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    # $common ./strips-ablation.py $mode lightsout twisted    4 {} "stopgrad" ::: 5000 ::: None ::: None ::: None ::: True  ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    # $common ./strips-ablation.py $mode hanoi     4 8          {} "stopgrad" ::: 5000 ::: None ::: None ::: None ::: True  ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    # $common ./strips-ablation.py $mode hanoi     9 3          {} "stopgrad" ::: 5000 ::: None ::: None ::: None ::: True  ::: ConcreteDetNormalizedLogitAddEffectTransitionAE

    # different arch
    # post-experiment note: significant.
    $common ./strips-ablation.py $mode puzzle    mnist    3 3 {} "arch" ::: 5000 ::: None ::: None ::: None ::: False ::: ConcreteDet{BoolMinMax,BoolSmoothMinMax,BoolAdd,LogitAdd,NormalizedLogitAdd}EffectTransitionAE
    $common ./strips-ablation.py $mode puzzle    spider   3 3 {} "arch" ::: 5000 ::: None ::: None ::: None ::: False ::: ConcreteDet{BoolMinMax,BoolSmoothMinMax,BoolAdd,LogitAdd,NormalizedLogitAdd}EffectTransitionAE
    $common ./strips-ablation.py $mode puzzle    mandrill 3 3 {} "arch" ::: 5000 ::: None ::: None ::: None ::: False ::: ConcreteDet{BoolMinMax,BoolSmoothMinMax,BoolAdd,LogitAdd,NormalizedLogitAdd}EffectTransitionAE
    $common ./strips-ablation.py $mode lightsout digital    4 {} "arch" ::: 5000 ::: None ::: None ::: None ::: False ::: ConcreteDet{BoolMinMax,BoolSmoothMinMax,BoolAdd,LogitAdd,NormalizedLogitAdd}EffectTransitionAE
    $common ./strips-ablation.py $mode lightsout twisted    4 {} "arch" ::: 5000 ::: None ::: None ::: None ::: False ::: ConcreteDet{BoolMinMax,BoolSmoothMinMax,BoolAdd,LogitAdd,NormalizedLogitAdd}EffectTransitionAE
    # $common ./strips-ablation.py $mode hanoi     4 8          {} "arch" ::: 5000 ::: None ::: None ::: None ::: False ::: ConcreteDet{BoolMinMax,BoolSmoothMinMax,BoolAdd,LogitAdd,NormalizedLogitAdd}EffectTransitionAE
    $common ./strips-ablation.py $mode hanoi     9 3          {} "arch" ::: 5000 ::: None ::: None ::: None ::: False ::: ConcreteDet{BoolMinMax,BoolSmoothMinMax,BoolAdd,LogitAdd,NormalizedLogitAdd}EffectTransitionAE

    # action label restriction, comparing Conditional and NormLogit
    # post-experiment note: not significant.
    # $common ./strips-ablation.py $mode puzzle    mnist    3 3 {} "actions" ::: 5000 ::: None ::: 50 400 ::: None ::: False ::: ConcreteDet{Conditional,NormalizedLogitAdd}EffectTransitionAE
    # $common ./strips-ablation.py $mode puzzle    spider   3 3 {} "actions" ::: 5000 ::: None ::: 50 400 ::: None ::: False ::: ConcreteDet{Conditional,NormalizedLogitAdd}EffectTransitionAE
    # $common ./strips-ablation.py $mode puzzle    mandrill 3 3 {} "actions" ::: 5000 ::: None ::: 50 400 ::: None ::: False ::: ConcreteDet{Conditional,NormalizedLogitAdd}EffectTransitionAE
    # $common ./strips-ablation.py $mode lightsout digital    4 {} "actions" ::: 5000 ::: None ::: 50 400 ::: None ::: False ::: ConcreteDet{Conditional,NormalizedLogitAdd}EffectTransitionAE
    # $common ./strips-ablation.py $mode lightsout twisted    4 {} "actions" ::: 5000 ::: None ::: 50 400 ::: None ::: False ::: ConcreteDet{Conditional,NormalizedLogitAdd}EffectTransitionAE
    # $common ./strips-ablation.py $mode hanoi     4 8          {} "actions" ::: 5000 ::: None ::: 50 400 ::: None ::: False ::: ConcreteDet{Conditional,NormalizedLogitAdd}EffectTransitionAE
    # $common ./strips-ablation.py $mode hanoi     9 3          {} "actions" ::: 5000 ::: None ::: 50 400 ::: None ::: False ::: ConcreteDet{Conditional,NormalizedLogitAdd}EffectTransitionAE

    # latent space restriction, comparing Conditional and NormLogit
    # post-experiment note: not significant.
    # $common ./strips-ablation.py $mode puzzle    mnist    3 3 {} "bits" ::: 5000 ::: 100 1000 ::: None ::: None ::: False ::: ConcreteDet{Conditional,NormalizedLogitAdd}EffectTransitionAE
    # $common ./strips-ablation.py $mode puzzle    spider   3 3 {} "bits" ::: 5000 ::: 100 1000 ::: None ::: None ::: False ::: ConcreteDet{Conditional,NormalizedLogitAdd}EffectTransitionAE
    # $common ./strips-ablation.py $mode puzzle    mandrill 3 3 {} "bits" ::: 5000 ::: 100 1000 ::: None ::: None ::: False ::: ConcreteDet{Conditional,NormalizedLogitAdd}EffectTransitionAE
    # $common ./strips-ablation.py $mode lightsout digital    4 {} "bits" ::: 5000 ::: 100 1000 ::: None ::: None ::: False ::: ConcreteDet{Conditional,NormalizedLogitAdd}EffectTransitionAE
    # $common ./strips-ablation.py $mode lightsout twisted    4 {} "bits" ::: 5000 ::: 100 1000 ::: None ::: None ::: False ::: ConcreteDet{Conditional,NormalizedLogitAdd}EffectTransitionAE
    # $common ./strips-ablation.py $mode hanoi     4 8          {} "bits" ::: 5000 ::: 100 1000 ::: None ::: None ::: False ::: ConcreteDet{Conditional,NormalizedLogitAdd}EffectTransitionAE
    # $common ./strips-ablation.py $mode hanoi     9 3          {} "bits" ::: 5000 ::: 100 1000 ::: None ::: None ::: False ::: ConcreteDet{Conditional,NormalizedLogitAdd}EffectTransitionAE
}

task-vanilla (){
    mode=$1
    # for Cube-AAE --- limit=100

    $common ./strips-vanilla.py $mode puzzle    mnist    3 3 {} "vanilla" ::: 5000 ::: None ::: None ::: None ::: False ::: VanillaTransitionAE
    $common ./strips-vanilla.py $mode puzzle    spider   3 3 {} "vanilla" ::: 5000 ::: None ::: None ::: None ::: False ::: VanillaTransitionAE
    $common ./strips-vanilla.py $mode puzzle    mandrill 3 3 {} "vanilla" ::: 5000 ::: None ::: None ::: None ::: False ::: VanillaTransitionAE
    $common ./strips-vanilla.py $mode lightsout digital    4 {} "vanilla" ::: 5000 ::: None ::: None ::: None ::: False ::: VanillaTransitionAE
    $common ./strips-vanilla.py $mode lightsout twisted    4 {} "vanilla" ::: 5000 ::: None ::: None ::: None ::: False ::: VanillaTransitionAE
    # $common ./strips-vanilla.py $mode hanoi     4 8          {} "vanilla" ::: 5000 ::: None ::: None ::: None ::: False ::: VanillaTransitionAE
    $common ./strips-vanilla.py $mode hanoi     9 3          {} "vanilla" ::: 5000 ::: None ::: None ::: None ::: False ::: VanillaTransitionAE
}

task-16puzzle (){
    mode=$1
    # main training for plannning --- limit=300
    
    $common ./strips-16.py $mode puzzle    mandrill 4 4 {} "16puzzle" ::: 50000 ::: None ::: None ::: None ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
}

task-fixedactions (){
    mode=$1
    actions=$2
    # main training for plannning --- limit=300

    # fixedactions version
    $common ./strips-ablation.py $mode puzzle    mnist    3 3 {} "fixedactions" ::: 5000 ::: None ::: $actions ::: None ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE 
    $common ./strips-ablation.py $mode puzzle    spider   3 3 {} "fixedactions" ::: 5000 ::: None ::: $actions ::: None ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    $common ./strips-ablation.py $mode puzzle    mandrill 3 3 {} "fixedactions" ::: 5000 ::: None ::: $actions ::: None ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    $common ./strips-ablation.py $mode lightsout digital    4 {} "fixedactions" ::: 5000 ::: None ::: $actions ::: None ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    $common ./strips-ablation.py $mode lightsout twisted    4 {} "fixedactions" ::: 5000 ::: None ::: $actions ::: None ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    # $common ./strips-ablation.py $mode hanoi     4 8          {} "fixedactions" ::: 5000 ::: None ::: $actions ::: None ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
    $common ./strips-ablation.py $mode hanoi     9 3          {} "fixedactions" ::: 5000 ::: None ::: $actions ::: None ::: False ::: ConcreteDetNormalizedLogitAddEffectTransitionAE
}

# for training the NN from the scratch using GA-based hyperparameter tuner (takes ~24hrs)

proj=$(date +%Y%m%d%H%M)sae-planning
common="parallel -j 1 --keep-order jbsub -mem 32g -cores 1+1 -queue x86_24h -proj $proj"
# task-planning      learn_plot_dump_summary

proj=$(date +%Y%m%d%H%M)sae-ablation
common="parallel -j 1 --keep-order jbsub -mem 32g -cores 1+1 -queue x86_24h -proj $proj"
# task-ablation     learn_summary

proj=$(date +%Y%m%d%H%M)sae-vanilla
common="parallel -j 1 --keep-order jbsub -mem 32g -cores 1+1 -queue x86_24h -proj $proj"
# task-vanilla      learn_plot_dump_summary

proj=$(date +%Y%m%d%H%M)sae-16puzzle
common="parallel -j 1 --keep-order jbsub -mem 32g -cores 1+1 -queue x86_6h -proj $proj"
# task-16puzzle     learn_plot_dump_summary

proj=$(date +%Y%m%d%H%M)sae-fixedactions
common="parallel -j 1 --keep-order jbsub -mem 32g -cores 1+1 -queue x86_6h -proj $proj"
# task-fixedactions     learn 300
# task-fixedactions     learn 200
# task-fixedactions     learn 100
# task-fixedactions     learn  50



# for training the NN 3 times with the best hyperparameter found in the result log (takes ~2hrs)

proj=$(date +%Y%m%d%H%M)reproduce-planning
common="parallel -j 1 --keep-order jbsub -mem 32g -cores 1+1 -queue x86_6h -proj $proj"
# task-planning      reproduce_plot_dump_summary

proj=$(date +%Y%m%d%H%M)reproduce-ablation
common="parallel -j 1 --keep-order jbsub -mem 32g -cores 1+1 -queue x86_6h -proj $proj"
# task-ablation     reproduce_summary

proj=$(date +%Y%m%d%H%M)reproduce-vanilla
common="parallel -j 1 --keep-order jbsub -mem 32g -cores 1+1 -queue x86_6h -proj $proj"
# task-vanilla      reproduce_plot_dump_summary

proj=$(date +%Y%m%d%H%M)reproduce-16puzzle
common="parallel -j 1 --keep-order jbsub -mem 32g -cores 1+1 -queue x86_6h -proj $proj"
# task-16puzzle     reproduce_plot_dump_summary

proj=$(date +%Y%m%d%H%M)sae-fixedactions
common="parallel -j 1 --keep-order jbsub -mem 32g -cores 1+1 -queue x86_6h -proj $proj"
# task-fixedactions     reproduce 300
# task-fixedactions     reproduce 200
# task-fixedactions     reproduce 100
# task-fixedactions     reproduce  50



# regenerate the summary from the stored weights

proj=$(date +%Y%m%d%H%M)dump-planning
common="parallel -j 1 --keep-order jbsub -mem 32g -cores 1+1 -queue x86_1h -proj $proj"
# task-planning      dump_plot_dump_summary

proj=$(date +%Y%m%d%H%M)dump-ablation
common="parallel -j 1 --keep-order jbsub -mem 32g -cores 1+1 -queue x86_1h -proj $proj"
# task-ablation     dump_summary

proj=$(date +%Y%m%d%H%M)dump-vanilla
common="parallel -j 1 --keep-order jbsub -mem 32g -cores 1+1 -queue x86_1h -proj $proj"
# task-vanilla      dump_plot_dump_summary

proj=$(date +%Y%m%d%H%M)dump-16puzzle
common="parallel -j 1 --keep-order jbsub -mem 32g -cores 1+1 -queue x86_1h -proj $proj"
# task-16puzzle     dump_plot_dump_summary

proj=$(date +%Y%m%d%H%M)dump-fixedactions
common="parallel -j 1 --keep-order jbsub -mem 32g -cores 1+1 -queue x86_1h -proj $proj"
# task-fixedactions     summary 300
# task-fixedactions     summary 200
# task-fixedactions     summary 100
# task-fixedactions     summary  50

#!/bin/bash

export dir=${1:-samples-ablation}
entries=".sae.MSE.vanilla.test .sae.MSE_aae.vanilla.test//.sae.aae.MSE.vanilla.test .sae.MAE_aae_z.vanilla.test//.sae.aae.z_MAE.vanilla.test .sae.MSE.vanilla.test+(.sae.MSE_aae.vanilla.test//.sae.aae.MSE.vanilla.test)+(.sae.MAE_aae_z.vanilla.test//.sae.aae.z_MAE.vanilla.test)"

run(){
    ./performance.sh $(ls -vd $dir/*/ | grep $1) -- $entries  | grep -e $1 -e directory
}

echo test

run DetNormalizedLogitAdd.*arch/
run DetLogitAdd.*arch/
run nodirect.*arch/
run NoSuc.*arch/
# run BoolAdd.*arch
run DetBoolMinMax.*arch/
run DetBoolSmooth.*arch/
# run Cond.*actions
# run Norm.*actions
# run Cond.*bits
# run Norm.*bits
# run stopgrad
# run precond/
# run precond2
# run precond3
# run longpre/
# run longpre2

# entries=".sae.MSE.vanilla.train .sae.MSE_aae.vanilla.train//.sae.aae.MSE.vanilla.train .sae.MAE_aae_z.vanilla.train//.sae.aae.z_MAE.vanilla.train .sae.MSE.vanilla.train+(.sae.MSE_aae.vanilla.train//.sae.aae.MSE.vanilla.train)+(.sae.MAE_aae_z.vanilla.train//.sae.aae.z_MAE.vanilla.train)"
# 
# echo train
# run DetNormalizedLogitAdd.*arch
# # run BoolAdd.*arch
# run DetBoolMinMax.*arch
# run DetBoolSmooth.*arch
# run DetLogitAdd.*arch
# # run Cond.*actions
# # run Norm.*actions
# # run Cond.*bits
# # run Norm.*bits
# run nodirect
# # run stopgrad
# # run precond/
# # run precond2
# # run precond3
# # run longpre/
# # run longpre2

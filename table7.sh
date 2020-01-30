#!/bin/bash

# Cube-space AE vs Cube AAE

fn (){
    mode=$1
    echo "### $mode ###"

    ./performance.sh samples-for-aae/*                 -- .nothing                       .nothing                    .nothing                          .sae.MSE.vanilla.${mode}
    ./performance.sh samples-for-aae/*/_Cube*None      -- .sae.mae.${mode}               .sae.parameters.M           .sae.effective_labels
    ./performance.sh samples-ablation/*DetNormal*arch  -- .sae.aae.z_MAE.vanilla.${mode} .sae.parameters.num_actions .sae.aae.true_num_actions.${mode} .sae.MSE.vanilla.${mode}
}

# this actually must be train, which contains a larger set
# fn test
fn train

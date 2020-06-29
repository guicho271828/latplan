#!/bin/bash -x

set -e

trap exit SIGINT

ulimit -v 16000000000

export PYTHONUNBUFFERED=1

base=samples-for-aae

proj=$(date +%Y%m%d%H%M)aae
common="jbsub -mem 32g -cores 1+1 -queue x86_12h -proj $proj"

# for training the NN from the scratch using GA-based hyperparameter tuner (100 iterations)

# parallel $common {} \
#          ::: ./action_autoencoder.py \
#          ::: $base/*/ \
#          ::: learn_test_dump \
#          ::: ActionAE CubeActionAE \
#          ::: None

# for training the NN 3 times with the best hyperparameter found in the result log

# parallel $common {} \
#          ::: ./action_autoencoder.py \
#          ::: $base/*/ \
#          ::: reproduce_test_dump \
#          ::: ActionAE CubeActionAE \
#          ::: None


# you must wait for the AAE jobs to finish before training sd and ad

# watch-proj $proj

proj=$(date +%Y%m%d%H%M)sd
common="jbsub -mem 32g -cores 1+1 -queue x86_6h -proj $proj"

# parallel $common {} \
#          ::: ./state_discriminator3.py \
#          ::: $base/*/ \
#          ::: learn_test \
#          ::: direct
# 


# depens on ad
proj=$(date +%Y%m%d%H%M)ad
common="jbsub -mem 32g -cores 1+1 -queue x86_6h -proj $proj"

# parallel $common {} \
#          ::: ./action_discriminator.py \
#          ::: $base/*/ \
#          ::: learn_test \
#          ::: _ActionAE_None _CubeActionAE_None

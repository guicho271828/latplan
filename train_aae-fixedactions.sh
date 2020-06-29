#!/bin/bash -x

a=$1

set -e

trap exit SIGINT

ulimit -v 16000000000

export PYTHONUNBUFFERED=1

base=samples-for-aae

proj=$(date +%Y%m%d%H%M)aae-fixedactions
common="jbsub -mem 32g -cores 1+1 -queue x86_12h -proj $proj"

# for training the NN from the scratch using GA-based hyperparameter tuner (100 iterations)

# parallel $common {} \
#          ::: ./action_autoencoder.py \
#          ::: $base/*/ \
#          ::: learn_test_dump \
#          ::: ActionAE CubeActionAE \
#          ::: $a

# for training the NN 3 times with the best hyperparameter found in the result log

# parallel $common {} \
#          ::: ./action_autoencoder.py \
#          ::: $base/*/ \
#          ::: reproduce_test_dump \
#          ::: ActionAE CubeActionAE \
#          ::: $a

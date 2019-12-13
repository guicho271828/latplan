#!/bin/bash -x

a=$1

set -e

trap exit SIGINT

ulimit -v 16000000000

export dir=$(dirname $(dirname $(readlink -ef $0)))
export PYTHONPATH=$dir:$PYTHONPATH
export PYTHONUNBUFFERED=1

base=samples-for-aae

proj=$(date +%Y%m%d%H%M)aae-fixedactions
common="jbsub -mem 32g -cores 1+1 -queue x86_12h -proj $proj"

parallel $common {} \
         ::: ./action_autoencoder.py \
         ::: $base/*/ \
         ::: learn_test_dump \
         ::: ActionAE CubeActionAE \
         ::: $a


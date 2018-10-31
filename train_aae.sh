#!/bin/bash -x

set -e

trap exit SIGINT

ulimit -v 16000000000

dir=$(dirname $(dirname $(readlink -ef $0)))
proj=$(date +%Y%m%d%H%M)
common="jbsub -mem 64g -cores 1+1 -queue x86_24h -proj $proj"

parallel $common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 {} \
         ::: ./action_autoencoder.py \
         ::: \
         samples/* \
         ::: learn_test_dump

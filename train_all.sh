#!/bin/bash

set -e

trap exit SIGINT

ulimit -v 16000000000

dir=$(dirname $(dirname $(readlink -ef $0)))
proj=$(date +%Y%m%d%H%M)sae
# Example command for lsf job scheduler
common="--keep-order -j 1 jbsub -mem 64g -cores 1+1 -queue x86_1h -proj $proj PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1"

parallel $common ./strips.py learn_plot_dump puzzle HammingTransitionAE mandrill 3 3 ::: 100 ::: 10000 ::: 0.0 0.5 ::: 0.0
parallel $common ./strips.py learn_plot_dump puzzle HammingTransitionAE mnist 3 3    ::: 100 ::: 10000 ::: 0.0 0.5 ::: 0.0
parallel $common ./strips.py learn_plot_dump puzzle HammingTransitionAE spider 3 3   ::: 100 ::: 10000 ::: 0.0 0.5 ::: 0.0

parallel $common ./strips.py learn_plot_dump lightsout HammingTransitionAE digital 4 ::: 100 ::: 10000 ::: 0.0 0.5 ::: 0.0
parallel $common ./strips.py learn_plot_dump lightsout HammingTransitionAE twisted 4 ::: 100 ::: 10000 ::: 0.0 0.5 ::: 0.0

# parallel $morecommon ./strips.py learn_plot_dump  hanoi HammingTransitionAE 4 4           ::: 100 1000 ::: 10000 ::: 0.0 0.5 ::: 0.0

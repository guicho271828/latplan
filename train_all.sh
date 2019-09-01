#!/bin/bash -x

set -e

trap exit SIGINT

ulimit -v 16000000000

dir=$(dirname $(dirname $(readlink -ef $0)))
proj=$(date +%Y%m%d%H%M)
common="jbsub -mem 64g -cores 1+1 -queue x86_1h -proj $proj"

morecommon="-j 1 --keep-order $common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 ./strips.py learn"

# parallel $morecommon puzzle {} mnist 3 3 10 10 0.0 "test" \
#          ::: \
#          {Concrete,Quantized,Sigmoid,MultiValued}{NonDetAction,DetAction,NonDetStrips,DetStrips1,DetStrips2}TransitionAE

common="jbsub -mem 64g -cores 1+1 -queue x86_12h -proj $proj"

morecommon="-j 1 --keep-order $common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 ./strips.py learn_plot_dump"

parallel $morecommon puzzle {} mnist 3 3 None 3000 0.0 "oct11" \
         ::: \
         {Concrete,Quantized,Sigmoid,MultiValued}{NonDet,Det,DiffDet}{Conditional,ConditionalStop,Logit,LogitStop}EffectTransitionAE

# ensure it fails

# parallel $morecommon puzzle StripsTransitionAE mnist 3 3    ::: 100 ::: 3000 ::: 0.0 ::: ""
# parallel $morecommon puzzle StripsTransitionAE spider 3 3   ::: 100 ::: 3000 ::: 0.7
# 
# parallel $morecommon lightsout StripsTransitionAE digital 4 ::: 100 ::: 3000 ::: 0.7
# parallel $morecommon lightsout StripsTransitionAE twisted 4 ::: 100 ::: 3000 ::: 0.7
# 
# parallel $morecommon hanoi StripsTransitionAE 4 4           ::: 100 ::: 3000 ::: 0.7 

echo
echo $proj
echo
info-proj $proj

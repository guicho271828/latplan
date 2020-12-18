#!/bin/bash

#!/bin/bash -x

set -e

trap exit SIGINT

ulimit -v 16000000000

proj=$(date +%Y%m%d%H%M)setup

# without job scheduler
common="PYTHONUNBUFFERED=1"

# with job scheduler
common="jbsub -mem 64g -cores 1 -queue x86_6h -proj $proj PYTHONUNBUFFERED=1"

# some instances clearly does not have 50000 states/transitiosn, but lets forget about it for now
$common ./setup-dataset.py puzzle mnist 3 3 50000
$common ./setup-dataset.py puzzle mnist 4 4 50000
$common ./setup-dataset.py puzzle mandrill 3 3 50000
$common ./setup-dataset.py puzzle mandrill 4 4 50000
$common ./setup-dataset.py puzzle spider 3 3 50000
$common ./setup-dataset.py puzzle spider 4 4 50000
$common ./setup-dataset.py lightsout digital 4 50000
$common ./setup-dataset.py lightsout digital 5 50000
$common ./setup-dataset.py lightsout twisted 4 50000
$common ./setup-dataset.py lightsout twisted 5 50000
$common ./setup-dataset.py hanoi 3 3 50000
$common ./setup-dataset.py hanoi 4 4 50000
$common ./setup-dataset.py hanoi 9 3 50000
$common ./setup-dataset.py hanoi 4 8 50000

download-and-extract (){
    wget https://github.com/IBM/photorealistic-blocksworld/releases/download/$1/$1.npz -O latplan/puzzles/$1.npz
    wget https://github.com/IBM/photorealistic-blocksworld/releases/download/$1/$1-init.json -O latplan/puzzles/$1-init.json
    wget https://github.com/IBM/photorealistic-blocksworld/releases/download/$1/$1-stat.json -O latplan/puzzles/$1-stat.json
}

export -f download-and-extract

$common download-and-extract blocks-5-3
$common download-and-extract blocks-4-4
$common download-and-extract blocks-3-7
$common download-and-extract blocks-3-6
$common download-and-extract blocks-3-5
$common download-and-extract blocks-3-4
$common download-and-extract blocks-3-3
$common download-and-extract blocks-3-3-multi
$common download-and-extract blocks-3-3-multi-global

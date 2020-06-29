#!/bin/bash

#!/bin/bash -x

set -e

trap exit SIGINT

ulimit -v 16000000000

proj=$(date +%Y%m%d%H%M)setup
common="jbsub -mem 64g -cores 1 -queue x86_1h -proj $proj"

# some instances clearly does not have 50000 states/transitiosn, but lets forget about it for now
$common PYTHONUNBUFFERED=1 ./setup.py puzzle mnist 3 3 50000
$common PYTHONUNBUFFERED=1 ./setup.py puzzle mnist 4 4 50000
$common PYTHONUNBUFFERED=1 ./setup.py puzzle mandrill 3 3 50000
$common PYTHONUNBUFFERED=1 ./setup.py puzzle mandrill 4 4 50000
$common PYTHONUNBUFFERED=1 ./setup.py puzzle spider 3 3 50000
$common PYTHONUNBUFFERED=1 ./setup.py puzzle spider 4 4 50000
$common PYTHONUNBUFFERED=1 ./setup.py lightsout digital 4 50000
$common PYTHONUNBUFFERED=1 ./setup.py lightsout digital 5 50000
$common PYTHONUNBUFFERED=1 ./setup.py lightsout twisted 4 50000
$common PYTHONUNBUFFERED=1 ./setup.py lightsout twisted 5 50000
$common PYTHONUNBUFFERED=1 ./setup.py hanoi 3 3 50000
$common PYTHONUNBUFFERED=1 ./setup.py hanoi 4 4 50000
$common PYTHONUNBUFFERED=1 ./setup.py hanoi 9 3 50000
$common PYTHONUNBUFFERED=1 ./setup.py hanoi 4 8 50000

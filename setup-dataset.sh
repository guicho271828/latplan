#!/bin/bash

#!/bin/bash -x

set -e

trap exit SIGINT

ulimit -v 16000000000

dir=$(dirname $(dirname $(readlink -ef $0)))
proj=$(date +%Y%m%d%H%M)
common="jbsub -mem 64g -cores 1 -queue x86_1h -proj $proj"

$common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 ./setup.py puzzle mnist 3 3
$common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 ./setup.py puzzle mnist 4 4
$common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 ./setup.py puzzle mandrill 3 3
$common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 ./setup.py puzzle mandrill 4 4
$common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 ./setup.py puzzle spider 3 3
$common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 ./setup.py puzzle spider 4 4
$common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 ./setup.py lightsout digital 4
$common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 ./setup.py lightsout digital 5
$common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 ./setup.py lightsout twisted 4
$common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 ./setup.py lightsout twisted 5

parallel $common PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1 ./setup.py hanoi ::: {3..6} ::: {3..8}

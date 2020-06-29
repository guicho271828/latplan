#!/bin/bash

probdir=${1:-problem-instances}

# common="-j 8 PYTHONPATH=$dir:$PYTHONPATH ./ood.py"
# common="--dry-run PYTHONPATH=$dir:$PYTHONPATH ./ood.py"

proj=$(date +%Y%m%d%H%M)ood
common="jbsub -mem 8g -cores 1 -queue x86_1h -proj $proj ./ood.py"

parallel $common puzzle    mnist    3 3 5000 {} {.}.json False ::: $probdir/*/latplan.puzzles.puzzle_mnist/*/*.npz
parallel $common puzzle    spider   3 3 5000 {} {.}.json False ::: $probdir/*/latplan.puzzles.puzzle_spider/*/*.npz
parallel $common puzzle    mandrill 3 3 5000 {} {.}.json False ::: $probdir/*/latplan.puzzles.puzzle_mandrill/*/*.npz
parallel $common lightsout digital    4 5000 {} {.}.json False ::: $probdir/*/latplan.puzzles.lightsout_digital/*/*.npz
parallel $common lightsout twisted    4 5000 {} {.}.json False ::: $probdir/*/latplan.puzzles.lightsout_twisted/*/*.npz
parallel $common hanoi     4 8          5000 {} {.}.json False ::: $probdir/*/latplan.puzzles.hanoi/*/*.npz
parallel $common hanoi     9 3          5000 {} {.}.json False ::: $probdir/*/latplan.puzzles.hanoi/*/*.npz
parallel $common puzzle    mandrill 4 4 5000 {} {.}.json True  ::: $probdir/*/latplan.puzzles.puzzle_mandrill/*/*.npz

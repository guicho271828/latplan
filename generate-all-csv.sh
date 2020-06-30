#!/bin/bash

# generate the test loss comparison for ablation study
./table4.sh > table4.csv
# generate the loss comparison for Cube-space AE vs Cube AAE
./table7.sh > table7.csv

# count the out-of-distribution states and transitions from the generated plans,
# and dump the results into json files
./ood.sh problem-instances
./ood.sh problem-instances-16
./ood.sh problem-instances-16-korf

# generate a histogram from the json files
./ood-histogram.sh problem-instances
./ood-histogram.sh problem-instances-16
./ood-histogram.sh problem-instances-16-korf

# generate a coverage table (coverage = number of instances solved)
./check1.sh problem-instances
./check1.sh problem-instances-16
./check1.sh problem-instances-16-korf

# generate a search statistics summary
./statistics.sh problem-instances
./statistics.sh problem-instances-16 true
# only for mands/lmcut, and korf is solved only by LAMA
# ./statistics.sh problem-instances-16-korf true

# generate figures
make

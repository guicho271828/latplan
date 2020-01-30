#!/bin/bash

./table4.sh > table4.csv
./table7.sh > table7.csv

./ood.sh problem-instances
./ood.sh problem-instances-16
./ood.sh problem-instances-16-korf

# wait for the jobs to finish

./ood-histogram.sh problem-instances
./ood-histogram.sh problem-instances-16
./ood-histogram.sh problem-instances-16-korf

./check1.sh problem-instances
./check1.sh problem-instances-16
./check1.sh problem-instances-16-korf

./statistics.sh problem-instances
./statistics.sh problem-instances-16 true
# only for mands/lmcut, and korf is solved only by LAMA
# ./statistics.sh problem-instances-16-korf true


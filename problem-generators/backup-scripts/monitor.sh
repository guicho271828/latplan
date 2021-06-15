#!/bin/bash

echo "printing the number of successful runs etc. run it like watch -d $0 to continue watching."

for d in $(dirname $0)/*/
do
    pddl=$(find $d -name "*.pddl" | wc -l)
    plan=$(find $d -name "*.plan"        | wc -l)
    nega=$(find $d -name "*.negative"    | wc -l)
    fail=$((pddl - plan - nega))
    echo $d
    echo pddl: $pddl
    echo plan: $plan
    echo nega: $nega
    echo fail: $fail
done

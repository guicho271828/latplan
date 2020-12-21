#!/bin/bash

set -e
set -x

./ama3-domain.ros test-ama3-actions.csv \
                  test-ama3-add.csv \
                  test-ama3-del.csv \
                  test-ama3-pos.csv \
                  test-ama3-neg.csv > ama3-domain.pddl

./ama3-problem.bin $(cat test-ama3-init.csv) $(cat test-ama3-goal.csv) > ama3-problem.pddl

../planner-scripts/limit.sh -t 900 -m 8000000 -- "../planner-scripts/fd-latest-clean -o '--search astar(lmcut())' -- ama3-problem.pddl ama3-domain.pddl"

arrival ama3-domain.pddl ama3-problem.pddl ama3-problem.plan ama3-problem.trace


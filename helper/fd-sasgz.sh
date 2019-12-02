#!/bin/bash +x

planner-scripts/limit.sh -- "planner-scripts/fd-sasgz-clean -o '$1' -- $2 $(dirname $2)/domain.pddl"

#!/bin/bash +x

planner-scripts/limit.sh -- "planner-scripts/probe-clean -- $2 $(dirname $2)/domain.pddl"

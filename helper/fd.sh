#!/bin/bash +x

planner-scripts/limit.sh -t 900 -m 2000000 -- "planner-scripts/fd-clean -o '$1' -- $2 $3"

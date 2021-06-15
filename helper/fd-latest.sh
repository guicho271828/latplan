#!/bin/bash +x

planner-scripts/limit.sh -t 600 -m 8000000 -- "planner-scripts/fd-latest-clean -o '$1' -- $2 $3"

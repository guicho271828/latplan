#!/bin/bash

planner-scripts/limit.sh -v -o "$1" -- fd-sasgz-clean $2 $(dirname $2)/domain.pddl

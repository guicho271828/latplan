#!/bin/bash

find $1 -name "*path*" -delete
find $1 -name "*reconstruction*" -delete
find $1 -name "*.negative" -delete
# find $1 -name "*sas*" -delete
find $1 -name "*.log" -delete
find $1 -name "*.err" -delete
find $1 -name "*.csv" -delete
find $1 -name "*lock" -delete
find $1 -name "*.pddl" -delete
find $1 -name "*.plan" -delete

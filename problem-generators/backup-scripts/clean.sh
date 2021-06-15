#!/bin/bash

echo "cleaning the log files and results in this directory"

find $1 -name "ama*" -delete

echo done!

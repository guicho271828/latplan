#!/bin/bash

cat $(find -name "*.csv") | sort -r | uniq > all.csv

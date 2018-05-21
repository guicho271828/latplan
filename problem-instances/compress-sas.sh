#!/bin/bash

find -name "*.sas" -or -name "*.sasp" | parallel gzip

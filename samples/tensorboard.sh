#!/bin/bash

while :
do
    tensorboard --logdir=$(echo $1*/logs/ | tr " " ",")
done

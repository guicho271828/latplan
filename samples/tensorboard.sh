#!/bin/bash

while :
do
    tensorboard --logdir=$(echo */logs/ | tr " " ",")
done

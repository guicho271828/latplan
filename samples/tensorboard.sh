#!/bin/bash

tensorboard --logdir=$(echo */logs/ | tr " " ",")

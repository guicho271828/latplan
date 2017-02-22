#!/usr/bin/env python3

import numpy as np

def generate_configs(size=10):
    assert size <= 10
    return list(range(size))

def successors(config,size):
    succ = []
    if config +1 < size:
        succ.append(config+1)
    if config -1 >= 0:
        succ.append(config-1)
    return succ


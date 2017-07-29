#!/usr/bin/env python3

import sys
sys.path.append('../../')

import latplan
from plot import puzzle_plot
puzzle_plot(latplan.puzzles.puzzle_wrong)

#!/usr/bin/env python3

import os
import os.path
d = os.path.dirname(__file__)
os.environ["FF_PATH"] = os.path.join(d, "../planner-scripts/ff")

import importlib
import numpy as np
import latplan
import pddlgym
import imageio
from skimage.transform import resize
from latplan.puzzles.sokoban import make_env, shrink, tile, plan_to_actions, validate_states, validate_transitions, default_layout_archive, load, archive, threshold


def sokoban_pddlgoal(i=0,test=False):
    # note: dynamic_action_space=True makes action_space pruned by preconditions
    env, successor = make_env(i, test)
    init_state, debug_info = env.reset()
    init_image = env.render(mode="human_crisp")

    import pddlgym
    plan = pddlgym.planning.run_planner(debug_info['domain_file'], debug_info['problem_file'], "ff")
    actions = plan_to_actions(env,init_state,plan)

    for action in actions:
        _, _, done, _ = env.step(action)
        print(action,done)

    goal_image = env.render(mode="human_crisp")

    imageio.imsave(os.path.join(d,f"sokoban_init_{i}_{test}.png"),init_image)
    imageio.imsave(os.path.join(d,f"sokoban_goal_{i}_{test}.png"),goal_image)

for i in range(5):
    sokoban_pddlgoal(i,False)

for i in range(4):
    sokoban_pddlgoal(i,True)


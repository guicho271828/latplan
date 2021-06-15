#!/usr/bin/env python3

import os
import os.path
os.environ["FF_PATH"] = os.path.join(os.path.dirname(__file__), "../planner-scripts/ff")

import importlib
import numpy as np
import latplan
import pddlgym
import imageio
from skimage.transform import resize
from latplan.puzzles.sokoban import make_env, shrink, tile, plan_to_actions, validate_states, validate_transitions, default_layout_archive, load, archive, threshold
from latplan.util.np_distances import mae, mse, inf
import latplan.util.stacktrace
load(default_layout_archive)

env, _ = make_env(2, False)

init_state, debug_info = env.reset()
init_image_original = env.render(mode="human_crisp")
init_image = shrink(init_image_original) # uint8, 0-255

# because of significant amount of shrinking (200x200 -> 4x4),
# there are so much numerical errors between the original data in TOKEN_IMAGES and panels, seemingly.

# imageio.imwrite("sokoban_init.png",init_image_original)
# imageio.imwrite("sokoban_init_shrink.png",init_image)
# imageio.imwrite("sokoban_clear_original.png",pddlgym.rendering.sokoban.TOKEN_IMAGES[pddlgym.rendering.sokoban.CLEAR])
# imageio.imwrite("sokoban_clear_resized.png",archive["panels"][pddlgym.rendering.sokoban.CLEAR])

# print("init_image_original_float",init_image_original)          # float 0-1
# print("init_image_original_int",init_image_original*256)  # float 0-255
# print("init_image_shrink_int",init_image)                            # int 0-255
# print("init_image_shrink_float_256",init_image/256)                  # float 0-1
# print("init_image_shrink_float_255",init_image/255)                  # float 0-1

# print(archive["panels"][pddlgym.rendering.sokoban.CLEAR])
# print("CLEAR",pddlgym.rendering.sokoban.TOKEN_IMAGES[pddlgym.rendering.sokoban.CLEAR])
# print("CLEAR",(archive["panels"][pddlgym.rendering.sokoban.CLEAR]))

print("CLEAR",mse(init_image[:tile,:tile]/255 , archive["panels"][pddlgym.rendering.sokoban.CLEAR]))
print("PLAYER",mse(init_image[:tile,:tile]/255 , archive["panels"][pddlgym.rendering.sokoban.PLAYER]))
print("WALL",mse(init_image[:tile,:tile]/255 , archive["panels"][pddlgym.rendering.sokoban.WALL]))
print("STONE",mse(init_image[:tile,:tile]/255 , archive["panels"][pddlgym.rendering.sokoban.STONE]))
print("GOAL",mse(init_image[:tile,:tile]/255 , archive["panels"][pddlgym.rendering.sokoban.GOAL]))
print("STONE_AT_GOAL",mse(init_image[:tile,:tile]/255 , archive["panels"][pddlgym.rendering.sokoban.STONE_AT_GOAL]))

print("build confusion matrix")
print("index: CLEAR, PLAYER, STONE, STONE_AT_GOAL, GOAL, WALL")
mat = np.zeros((pddlgym.rendering.sokoban.NUM_OBJECTS,
                pddlgym.rendering.sokoban.NUM_OBJECTS))
for i in range(pddlgym.rendering.sokoban.NUM_OBJECTS):
    for j in range(pddlgym.rendering.sokoban.NUM_OBJECTS):
        # panel-wise errors by L-inf norm
        mat[i,j] = inf(archive["panels"][i],archive["panels"][j])
print(mat)
print(mat < threshold)


images = [init_image]

plan = pddlgym.planning.run_planner(debug_info['domain_file'], debug_info['problem_file'], "ff")
actions = plan_to_actions(env,init_state,plan)

for action in actions:
    _, _, done, _ = env.step(action)
    images.append(shrink(env.render(mode="human_crisp"))) # uint8, 0-255
    print(action,done)

images = np.stack(images) / 255           # float, 0-1

try:
    print("validating states")
    print(validate_states(images))
    print("validating transitions")
    print(validate_transitions([images[:-1],images[1:]], check_states=False))
except:
    latplan.util.stacktrace.format()

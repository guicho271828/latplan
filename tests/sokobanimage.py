#!/usr/bin/env python3

import tqdm
import gym
import pddlgym
import imageio
import os
import os.path
from skimage.transform import resize

env = gym.make("PDDLEnvSokoban-v0")
obs, _ = env.reset()
for i in tqdm.tqdm(range(20)):
    action = env.action_space.sample(obs)
    obs, _, done, _ = env.step(action)

    if not os.path.exists("sokoban"):
        os.mkdir("sokoban")
    tile = 16

    image = env.render(mode="human")[2:-2,2:-2,:3]
    imageio.imwrite(f"sokoban/large-image{i}.png", image)
    image = resize(image, [tile*5,tile*5,3], preserve_range=True)
    imageio.imwrite(f"sokoban/small-image{i}.png", image)

    image = env.render(mode="egocentric")[2:-2,2:-2,:3]
    imageio.imwrite(f"sokoban/large-ego{i}.png", image)
    image = resize(image, [tile*5,tile*5,3], preserve_range=True)
    imageio.imwrite(f"sokoban/small-ego{i}.png", image)

    image = env.render(mode="human_crisp")
    imageio.imwrite(f"sokoban/crisp-large-image{i}.png", image)
    image = resize(image, [tile*5,tile*5,3], preserve_range=True)
    imageio.imwrite(f"sokoban/crisp-small-image{i}.png", image)

    image = env.render(mode="egocentric_crisp")
    imageio.imwrite(f"sokoban/crisp-large-ego{i}.png", image)
    image = resize(image, [tile*5,tile*5,3], preserve_range=True)
    imageio.imwrite(f"sokoban/crisp-small-ego{i}.png", image)

    if done:
        obs, _ = env.reset()


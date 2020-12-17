#!/usr/bin/env python3

import tqdm
import imageio
import os
import os.path
import numpy as np
import latplan
import importlib

# egocentric, image, object
with np.load(os.path.join(latplan.__path__[0],"puzzles","sokoban-10-True-False-True"+".npz")) as data:
    pres = data["pres"]
    print(pres.shape,pres.dtype,pres.max(),pres.min())
    pres = pres.reshape([*pres.shape[:2],16,16,3])
    if not os.path.isdir("sokobantest"):
        os.makedirs("sokobantest")
    for i in range(3):
        for j, patch in enumerate(pres[i]):       # for each object
            imageio.imwrite(f"sokobantest/{i}-{j}.png", patch)

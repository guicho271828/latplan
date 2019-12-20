
def ensure_directory(directory):
    if directory[-1] is "/":
        return directory
    else:
        return directory+"/"

sae = None
problem_dir = None
network_dir = None
ama_version = None

import os.path
def problem(path):
    return os.path.join(problem_dir,path)

def network(path):
    root, ext = os.path.splitext(path)
    return "{}_{}{}".format(ensure_directory(network_dir).split("/")[-2], root, ext)

def ama(path):
    root, ext = os.path.splitext(path)
    return "{}_{}{}".format(ama_version, root, ext)

def init_goal_misc(p):
    import imageio
    import numpy as np
    from .plot         import plot_grid
    from .np_distances import bce, mae, mse
    from ..puzzles.util import preprocess, normalize
    # is already enhanced, equalized
    init_image = normalize(imageio.imread(problem("init.png")))
    goal_image = normalize(imageio.imread(problem("goal.png")))
    init = sae.encode(np.expand_dims(init_image,0))[0].round().astype(int)
    goal = sae.encode(np.expand_dims(goal_image,0))[0].round().astype(int)
    
    print("init:",init_image.min(),init_image.max(),)
    print("goal:",goal_image.min(),goal_image.max(),)
    print(init)
    print(goal)
    rec = sae.decode(np.array([init,goal]))
    init_rec, goal_rec = rec
    print("init (reconstruction):",init_rec.min(),init_rec.max(),)
    print("goal (reconstruction):",goal_rec.min(),goal_rec.max(),)

    def r(i):
        s = i.shape
        return i.reshape((s[0]//2, 2, s[1]//2, 2)).mean(axis=(1,3))
    
    plot_grid([init_image,init_rec,init_image-init_rec,(init_image-init_rec).round(),
               init_image.round(),init_rec.round(),init_image.round()-init_rec.round(),(init_image.round()-init_rec.round()).round(),
               r(init_image),r(init_rec),r(init_image)-r(init_rec),(r(init_image)-r(init_rec)).round(),
               # r(init_image).round(),r(init_rec).round(),r(init_image).round()-r(init_rec).round(),(r(init_image).round()-r(init_rec).round()).round(),
               
               goal_image,goal_rec,goal_image-goal_rec,(goal_image-goal_rec).round(),
               goal_image.round(),goal_rec.round(),goal_image.round()-goal_rec.round(),(goal_image.round()-goal_rec.round()).round(),
               r(goal_image),r(goal_rec),r(goal_image)-r(goal_rec),(r(goal_image)-r(goal_rec)).round(),
               # r(goal_image).round(),r(goal_rec).round(),r(goal_image).round()-r(goal_rec).round(),(r(goal_image).round()-r(goal_rec).round()).round(),
               ],
              w=4,
              path=problem(ama(network("init_goal_reconstruction.png"))),verbose=True)

    import sys
    print("init BCE:",bce(init_image,init_rec))
    print("init MAE:",mae(init_image,init_rec))
    print("init MSE:",mse(init_image,init_rec))
    # if image_diff(init_image,init_rec) > image_threshold:
    #     print("Initial state reconstruction failed!")
    #     sys.exit(3)
    print("goal BCE:",bce(goal_image,goal_rec))
    print("goal MAE:",mae(goal_image,goal_rec))
    print("goal MSE:",mse(goal_image,goal_rec))
    # if image_diff(goal_image,goal_rec) > image_threshold:
    #     print("Goal state reconstruction failed!")
    #     sys.exit(4)
    if not np.all(p.validate_states(rec)):
        print("Init/Goal state reconstruction failed!")
        # sys.exit(3)
        print("But we continue anyways...")
    return init, goal

def setup_planner_utils(_sae, _problem_dir, _network_dir, _ama_version):
    global sae, problem_dir, network_dir, ama_version
    sae, problem_dir, network_dir, ama_version = \
        _sae, _problem_dir, _network_dir, _ama_version
    return

import subprocess
def echodo(cmd,*args,**kwargs):
    print(cmd,flush=True)
    subprocess.run(cmd,*args,**kwargs)

def echo_out(cmd):
    print(cmd)
    return subprocess.check_output(cmd)

import time
start = time.time()
times = [(0,0,"init")]
def log(message):
    now = time.time()
    wall = now-start
    elap = wall-times[-1][0]
    times.append((wall,elap,message))
    print("@[{: =10.3f} +{: =10.3f}] {}".format(wall,elap,message))


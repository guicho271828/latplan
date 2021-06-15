
def ensure_directory(directory):
    if directory[-1] == "/":
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
    return "{}_{}{}".format(network_dir.replace("/","_"), root, ext)

def ama(path):
    root, ext = os.path.splitext(path)
    return "{}_{}{}".format(ama_version, root, ext)

def init_goal_misc(p, cycle=1, noise=None):
    # import sys
    import imageio
    import numpy as np
    from .plot         import plot_grid
    from .np_distances import bce, mae, mse
    from .noise        import gaussian

    def load_image(name):
        image = imageio.imread(problem(f"{name}.png")) / 255
        if len(image.shape) == 2:
            image = image.reshape(*image.shape, 1)
        image = sae.output.normalize(image)
        return image

    def autoencode_image(name,image):
        state = sae.encode(np.array([image]))[0].round().astype(int)
        image_rec = sae.decode(np.array([state]))[0]
        print(f"{name} (input) min:",image.min(),"max:",image.max(),)
        print(f"{name} (recon) min:",image_rec.min(),"max:",image_rec.max(),)
        # print(f"{name} BCE:",bce(image,image_rec))
        # print(f"{name} MAE:",mae(image,image_rec))
        print(f"{name} MSE:",mse(image,image_rec))
        # print(state)
        # if image_diff(image,image_rec) > image_threshold:
        #     print("Initial state reconstruction failed!")
        #     sys.exit(3)
        return state, image_rec

    def load_and_encode_image(name):
        image0 = load_image(name)
        if noise is not None:
            print(f"adding gaussian noise N(0,{noise})")
            image = gaussian(image0, noise)
        else:
            image = image0
        images = [image]
        for i in range(cycle):
            state, image = autoencode_image(name,image)
            images.append(image)
        return image0, state, image, images

    init_image, init, init_rec, init_images = load_and_encode_image("init")
    goal_image, goal, goal_rec, goal_images = load_and_encode_image("goal")

    sae.plot(np.concatenate([init_images,goal_images]),
             path=problem(ama(network(f"init_goal_reconstruction.{cycle}.png"))))

    if p and not np.all(
            p.validate_states(
                np.squeeze(     # remove the channel dimension in monochrome domains
                    sae.render(
                        np.stack(
                            [init_rec,goal_rec]))))):
        print("Init/Goal state reconstruction failed!")
        # sys.exit(3)
        print("But we continue anyways...")
    return init, goal

def setup_planner_utils(_sae, _problem_dir, _network_dir, _ama_version):
    global sae, problem_dir, network_dir, ama_version
    sae, problem_dir, network_dir, ama_version = \
        _sae, _problem_dir, _network_dir, _ama_version
    return


def puzzle_module(sae):
    import importlib
    assert "generator" in sae.parameters
    p = importlib.import_module(sae.parameters["generator"])
    p.setup()
    return p


import subprocess
def echodo(cmd,*args,**kwargs):
    print(cmd,flush=True)
    subprocess.run(cmd,*args,**kwargs)

def echo_out(cmd):
    print(cmd)
    return subprocess.check_output(cmd)

import time
start = time.time()
times = [{"message":"init","wall":0,"elapsed":0}]
def log(message):
    now = time.time()
    wall = now-start
    elap = wall-times[-1]["wall"]
    times.append({"message":message,"wall":wall,"elapsed":elap})
    print("@[{: =10.3f} +{: =10.3f}] {}".format(wall,elap,message))


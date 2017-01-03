#!/usr/bin/env python

import numpy as np
import subprocess

def plot_grid(images,name="plan.png"):
    import matplotlib.pyplot as plt
    l = len(images)
    w = 6
    h = max(l//6,1)
    plt.figure(figsize=(20, h*2))
    for i,image in enumerate(images):
        # display original
        ax = plt.subplot(h,w,i+1)
        plt.imshow(image.reshape(28, 28),interpolation='nearest',cmap='gray',)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(name)

def echodo(cmd):
    subprocess.call(["echo"]+cmd)
    subprocess.call(cmd)
    
def latent_plan(init,goal,ae):
    ig_image = np.array([init,goal])
    ig_z     = ae.encode_binary(ig_image)
    ig_b     = np.round(ig_z)
    print ig_z
    if np.equal(ig_z[0],ig_z[1]).all():
        raise ValueError("same init/goal")
    plot_grid(
        np.concatenate(
            (ig_image,
             ae.decode_binary(ig_z),
             ae.decode_binary(ig_b)),
            axis=0),
        ae.local("initial_and_goal_states.png"))

    # start planning
    
    echodo(["lisp/pddl.ros",ae.local("actions.csv")] +
           list(ig_b.flatten().astype('int').astype('string')))
    echodo(["planner-scripts/limit.sh","-v","--","fd-clean",
            ae.local("problem.pddl"),
            ae.local("domain.pddl")])
    out = subprocess.check_output(["lisp/parse-plan.ros",ae.local("problem.plan")])
    lines = out.split("\n")
    numbers = np.array([ [ int(s) for s in l.split() ] for l in lines ])
    print numbers
    states = np.concatenate((numbers[0:1,0:latent_dim], numbers[:,latent_dim:latent_dim*2]))
    print states
    plan_images = ae.decode_binary(states)
    plot_grid(plan_images,ae.local('plan.png'))

if __name__ == '__main__':

    from model import GumbelAE
    ae = GumbelAE("samples/counter_model/")
    
    from mnist import mnist
    x_train,y_train, x_test,y_test = mnist()
    while True:
        try:
            import random
            latent_plan(random.choice(x_test),
                        random.choice(x_test),
                        ae)
            break
        except ValueError as e:
            print e

#!/usr/bin/env python

import numpy as np
import subprocess

def plot_grid(images,name="plan.png"):
    import matplotlib.pyplot as plt
    l = len(images)
    w = 5
    h = l//5+1
    plt.figure(figsize=(20, h*2))
    for i,image in enumerate(images):
        # display original
        ax = plt.subplot(h,w,i+1)
        plt.imshow(image,interpolation='nearest',cmap='gray',)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(name)

def echodo(cmd):
    subprocess.call(["echo"]+cmd)
    subprocess.call(cmd)

class SameValueError:
    pass
class NoPlanFound:
    pass
class NoInterestingPlanFound:
    pass
    
def latent_plan(init,goal,shape,ae):
    ig_x     = np.array([init,goal])
    ig_z     = ae.encode_binary(ig_x)
    ig_y     = ae.decode_binary(ig_z)
    ig_b     = np.round(ig_z)
    ig_by    = ae.decode_binary(ig_b)
    print ig_b
    if np.equal(ig_b[0],ig_b[1]).all():
        raise SameValueError()
    images = []
    for x,z,y,b,by in zip(ig_x, ig_z, ig_y, ig_b, ig_by):
        images.append(x.reshape(shape))
        images.append(z.reshape((4,4)))
        images.append(y.reshape(shape))
        images.append(b.reshape((4,4)))
        images.append(by.reshape(shape))
    plot_grid(images, ae.local("init_goal.png"))

    # start planning
    
    echodo(["lisp/pddl.ros",ae.local("actions.csv")] +
           list(ig_b.flatten().astype('int').astype('string')))
    echodo(["planner-scripts/limit.sh","-v","--","fd-clean",
            ae.local("problem.pddl"),
            ae.local("domain.pddl")])
    try:
        out = subprocess.check_output(["lisp/parse-plan.ros",ae.local("problem.plan")])
        lines = out.split("\n")
        if len(lines) is 2:
            raise NoInterestingPlanFound()
        numbers = np.array([ [ int(s) for s in l.split() ] for l in lines ])
        print numbers
        latent_dim = numbers.shape[1]/2
        states = np.concatenate((numbers[0:1,0:latent_dim],
                                 numbers[:,latent_dim:]))
        print states
        plan_images = ae.decode_binary(states)
        plot_grid(plan_images.reshape((-1,)+shape),ae.local('plan.png'))
    except subprocess.CalledProcessError:
        raise NoPlanFound()

if __name__ == '__main__':
    def plan_random(ae,shape,transitions):
        while True:
            try:
                import random
                latent_plan(random.choice(transitions[0]),
                            random.choice(transitions[0]),
                            shape,
                            ae)
                break
            except SameValueError as e:
                print e
            except NoPlanFound as e:
                print e
            except NoInterestingPlanFound as e:
                print e
    
    from model import GumbelAE
    import counter
    plan_random(GumbelAE("samples/counter_model/"),
                (28,28),
                counter.transitions(n=1000))
    import puzzle
    plan_random(GumbelAE("samples/puzzle_model/"),
                (12,10),
                puzzle.transitions(2,2))
    import mnist_puzzle
    plan_random(GumbelAE("samples/mnist_puzzle_model/"),
                (56,56),
                mnist_puzzle.transitions(2,2))
    import puzzle
    plan_random(GumbelAE("samples/puzzle3_model/"),
                (6*2,5*3),
                puzzle.transitions(3,2))
    

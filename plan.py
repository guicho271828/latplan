#!/usr/bin/env python

import numpy as np
import subprocess

from plot import plot_grid, plot_grid2, plot_ae

def echodo(cmd):
    subprocess.call(["echo"]+cmd)
    subprocess.call(cmd)

class SameValueError:
    pass
class NoPlanFound:
    pass
class NoInterestingPlanFound:
    pass
    
def latent_plan(init,goal,ae):
    ig_x, ig_z, ig_y, ig_b, ig_by = plot_ae(ae,np.array([init,goal]),"init_goal.png")

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
        plot_grid(plan_images,path=ae.local('plan.png'))
    except subprocess.CalledProcessError:
        raise NoPlanFound()

if __name__ == '__main__':
    def plan_random(ae,transitions):
        while True:
            try:
                import random
                latent_plan(random.choice(transitions[0]),
                            random.choice(transitions[0]),
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
    # plan_random(GumbelAE("samples/counter_model/"),
    #             counter.transitions(n=1000))
    # import puzzle
    # plan_random(GumbelAE("samples/puzzle_model/"),
    #             puzzle.transitions(2,2))
    # import mnist_puzzle
    # plan_random(GumbelAE("samples/mnist_puzzle_model/"),
    #             mnist_puzzle.transitions(2,2))
    # import puzzle
    # plan_random(GumbelAE("samples/puzzle3_model/"),
    #             puzzle.transitions(3,2))
    # import puzzle
    # plan_random(GumbelAE("samples/puzzle32p_model/"),
    #             puzzle.transitions(3,2))
    import mnist_puzzle
    plan_random(GumbelAE("samples/mnist_puzzle32_model/"),
                mnist_puzzle.transitions(3,2))
    import mnist_puzzle
    plan_random(GumbelAE("samples/mnist_puzzle32p_model/"),
                mnist_puzzle.transitions(3,2))
    
    

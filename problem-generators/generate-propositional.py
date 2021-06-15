#!/usr/bin/env python3

import random
import numpy as np
import imageio
import os
import os.path
import sys
import importlib
import subprocess
import datetime
sys.path.append('../../')
from latplan.util import curry
from latplan.util.noise import gaussian
from latplan.util.search import dijkstra, reservoir_sampling, untuple

def noise(fn, param, domain, *args):
    noise_fns.append(lambda a: fn(a,param))
    output_dirs.append(fn.__name__)
    domain(*args)

def identity(x):
    return x

steps     = 5
instances = 100
noise_fns     = [identity]
output_dirs = ["vanilla"]


def generate(name, ics, gcs, render_fn):
    inits = render_fn(ics)
    goals = render_fn(gcs)
    for noise_fn,output_dir in zip(noise_fns,output_dirs):
        inits = noise_fn(inits)
        goals = noise_fn(goals)
        for i,(init,goal) in enumerate(zip(inits,goals)):
            d = "{}/{}/{:03d}-{:03d}".format(output_dir,name,steps,i)
            if os.path.isdir(d):
                subprocess.call(["mv",d,d+"_old_"+datetime.datetime.today().isoformat()])
            os.makedirs(d)
            print(d)
            imageio.imsave(os.path.join(d,"init.png"),init)
            imageio.imsave(os.path.join(d,"goal.png"),goal)


################################################################

# puzzle domain where the goal state is a completed puzzle
def puzzle(type='mnist', width=3, height=3):
    p = importlib.import_module('latplan.puzzles.puzzle_{}'.format(type))
    p.setup()
    def successor_fn(config):
        r = p.successors(config,width,height)
        return [tuple(e) for e in r]

    def goal_state():
        config = np.arange(width*height)
        return tuple(config)

    goal = goal_state()
    init_candidates = untuple(dijkstra(goal, steps, successor_fn))
    ics = reservoir_sampling(init_candidates, instances)
    gcs = [ goal for i in range(len(ics))] # note: not range(instances), because ics may be shorter depending on search depth
    generate("-".join(["puzzle",type,str(width),str(height)]), ics, gcs, lambda configs: p.generate(np.array(configs),width,height))

# puzzle domain where the goal state is random
def puzzle_random_goal(type='mnist', width=3, height=3):
    p = importlib.import_module('latplan.puzzles.puzzle_{}'.format(type))
    p.setup()
    def successor_fn(config):
        r = p.successors(config,width,height)
        return [tuple(e) for e in r]

    def goal_state():
        config = np.arange(width*height)
        np.random.shuffle(config)
        return tuple(config)

    gcs = [ goal_state() for i in range(instances)]
    ics = [ reservoir_sampling(untuple(dijkstra(goal, steps, successor_fn)), 1)[0] for goal in gcs ]
    generate("-".join(["puzzle_random_goal",type,str(width),str(height)]), ics, gcs, lambda configs: p.generate(np.array(configs),width,height))


# puzzle instances for various known longest properties (longest optimal solution, most solutions, etc)
def puzzle_longest(type='mnist', width=3, height=3):
    assert width==3
    assert height==3
    p = importlib.import_module('latplan.puzzles.puzzle_{}'.format(type))
    p.setup()
    def goal_state():
        config = np.arange(width*height)
        return tuple(config)
    ics = [
        # from Reinfield '93
        # [8,0,6,5,4,7,2,3,1], # the second instance with the longest optimal solution 31
        [3,5,6,8,4,7,2,1,0],
        # [8,7,6,0,4,1,2,5,3], # the first instance with the longest optimal solution 31
        [1,8,6,7,4,3,2,5,0],
        # [8,5,6,7,2,3,4,1,0], # the first instance with the most solutions
        [8,7,4,5,6,1,2,3,0],
        # [8,5,4,7,6,3,2,1,0], # the second instance with the most solutions
        [8,7,6,5,2,1,4,3,0],
        # [8,6,7,2,5,4,3,0,1], # the "wrong"? hardest eight-puzzle from
        [7,8,3,6,5,4,1,2,0],
        # [6,4,7,8,5,0,3,2,1], # w01fe.com/blog/2009/01/the-hardest-eight-puzzle-instances-take-31-moves-to-solve/
        [5,8,7,6,1,4,0,2,3],
    ]
    goal = goal_state()
    gcs = [ goal for i in range(len(ics))]
    generate("-".join(["puzzle_longest",type,str(width),str(height)]), ics, gcs, lambda configs: p.generate(np.array(configs),width,height))


# hanoi where the goal state is a state with all disks on the right
def hanoi(disks=5, towers=3):
    p = importlib.import_module('latplan.puzzles.hanoi')
    p.setup()
    def successor_fn(config):
        r = p.successors(config,disks,towers)
        return [tuple(e) for e in r]

    def goal_state():
        config = np.full(disks,towers-1,dtype=int)
        return tuple(config)

    goal = goal_state()
    init_candidates = untuple(dijkstra(goal, steps, successor_fn))
    ics = reservoir_sampling(init_candidates, instances)
    gcs = [ goal for i in range(len(ics))] # note: not range(instances), because ics may be shorter depending on search depth
    generate("-".join(["hanoi",str(disks),str(towers)]), ics, gcs, lambda configs: p.generate(np.array(configs),disks,towers))


# lightsout with random goal state.
def hanoi_random_goal(disks=5, towers=3):
    p = importlib.import_module('latplan.puzzles.hanoi')
    p.setup()
    def successor_fn(config):
        r = p.successors(config,disks,towers)
        return [tuple(e) for e in r]

    def goal_state():
        config = np.random.randint(0,towers,disks,dtype=int)
        return tuple(config)

    gcs = [ goal_state() for i in range(instances)]
    ics = [ reservoir_sampling(untuple(dijkstra(goal, steps, successor_fn)), 1)[0] for goal in gcs ]
    generate("-".join(["hanoi_random_goal",str(disks),str(towers)]), ics, gcs, lambda configs: p.generate(np.array(configs),disks,towers))


# lightsout. due to large branching factor, generation with 14 steps takes 1 hours
def lightsout(type='digital', size=4):
    p = importlib.import_module('latplan.puzzles.lightsout_{}'.format(type))
    p.setup()
    def successor_fn(config):
        r = p.successors(config)
        return [tuple(e) for e in r]

    def goal_state():
        config = np.full((size*size),-1)
        return tuple(config)

    goal = goal_state()
    init_candidates = untuple(dijkstra(goal, steps, successor_fn))
    ics = reservoir_sampling(init_candidates, instances)
    gcs = [ goal for i in range(len(ics))] # note: not range(instances), because ics may be shorter depending on search depth
    generate("-".join(["lightsout",type,str(size)]), ics, gcs, lambda configs: p.generate(np.array(configs)))


# lightsout with random goal state. it is even less practical than the original version to generate many problems with this function
def lightsout_random_goal(type='digital', size=4):
    p = importlib.import_module('latplan.puzzles.lightsout_{}'.format(type))
    p.setup()
    def successor_fn(config):
        r = p.successors(config)
        return [tuple(e) for e in r]

    def goal_state():
        config = np.random.randint(0,2,size*size,dtype=int)-1
        return tuple(config)

    gcs = [ goal_state() for i in range(instances)]
    ics = [ reservoir_sampling(untuple(dijkstra(goal, steps, successor_fn)), 1)[0] for goal in gcs ]
    generate("-".join(["lightsout_random_goal",type,str(size)]), ics, gcs, lambda configs: p.generate(np.array(configs)))


# a variant of lightsout generator which explots order-invariant nature of this domain.
# XXX do not use, still results in suboptimal paths???
def lightsout_special(init_c,length,successor_fn):
    # generating lightout with dijkstra is extremely memory-demanding, as each node has 25 successors.
    # however, lightsout plans are order-invariant, so we can sample instances easily

    leds = len(init_c)

    import itertools
    for plan in itertools.combinations(range(leds), length):
        current = tuple(init_c)
        for action in plan:
            current = successor_fn(current)[action]
        print(current,plan)
        yield current


################################################################

from latplan.puzzles.sokoban import shrink, make_env, plan_to_actions

def sokoban(i=0,test=False):
    # note: dynamic_action_space=True makes action_space pruned by preconditions
    env, successor = make_env(i, test)
    init, _ = env.reset()

    goal_candidates = untuple(dijkstra(init, steps, successor))
    gcs = reservoir_sampling(goal_candidates, instances)
    ics = [ init for i in range(len(gcs))] # note: not range(instances), because ics may be shorter depending on search depth
    def render(state):
        env.set_state(state)
        return shrink(env.render(mode="human_crisp"))

    generate("-".join(["sokoban",str(i),str(test)]), ics, gcs, lambda states: map(render,states))


def sokoban_pddlgoal(i=0,test=False):
    # note: dynamic_action_space=True makes action_space pruned by preconditions
    env, successor = make_env(i, test)
    init_state, debug_info = env.reset()
    ics = [ init_state ]

    import pddlgym
    plan = pddlgym.planning.run_planner(debug_info['domain_file'], debug_info['problem_file'], "ff")
    actions = plan_to_actions(env,init_state,plan)

    for action in actions:
        goal_state, _, done, _ = env.step(action)
        print(action,done)

    gcs = [ goal_state ]

    def render(state):
        env.set_state(state)
        return shrink(env.render(mode="human_crisp"))

    generate("-".join(["sokoban_pddl",str(i),str(test)]), ics, gcs, lambda states: map(render,states))



################################################################

def main():
    try:
        print('args:',sys.argv)
        def myeval(str):
            try:
                return eval(str)
            except:
                return str
        
        global steps, instances
        steps = myeval(sys.argv[1])
        instances = myeval(sys.argv[2])
        task      = myeval(sys.argv[3])
    except:
        print(sys.argv[0], 'steps','instances','task','[task-specific args...]')
    task(*map(myeval,sys.argv[4:]))

if __name__ == '__main__':
    try:
        main()
    except:
        import latplan.util.stacktrace
        latplan.util.stacktrace.format()


import os.path
import numpy as np
import gym
import pddlgym
from .util import wrap


# returns an archive name from given keywords
def archive_path(type = "image", limit = 1000, egocentric = False, objects = True, stage=0, test=False):
    list = [f"sokoban_{type}",limit,
            ("egocentric" if egocentric else "global"),
            ("object"     if objects    else "global"),
            stage,
            ("test" if test else "train"),]
    return os.path.join(os.path.dirname(__file__),"-".join(map(str,list))+".npz")


# instantiates a gym environment and defines a successor function
def make_env(problem_id, test):
    if test:
        env = gym.make("PDDLEnvSokobanTest-v0")
    else:
        env = gym.make("PDDLEnvSokoban-v0")
    # pddlgym.utils.run_planning_demo(env,"ff")
    env.fix_problem_index(problem_id)

    def successor(obs):
        env.set_state(obs)
        for action in env.action_space.all_ground_literals(obs, valid_only=True):
            env.set_state(obs)
            obs2, _, _, _ = env.step(action)
            yield obs2

    return env, successor


# original tile size is 16x16; shrink this to 4x4
original_tile_size = 16
shrink_factor = 4
tile = original_tile_size // shrink_factor

from skimage.transform import resize
def shrink(x):
    x = x[:,:,:3]
    H, W, _ = x.shape
    x = resize(x, (H//shrink_factor,W//shrink_factor,3))
    x = x * 256
    x = x.astype("uint8")
    return x


# reachability analysis: return a boolean matrix whose cells are True when
# the cell is reachable by the player
def compute_reachability(wall,player):
    h, w = wall.shape

    wall2     = np.zeros((h+2,w+2),dtype=bool)
    reachable = np.zeros((h+2,w+2),dtype=bool)
    wall2[1:h+1,1:w+1] = wall
    reachable[1:h+1,1:w+1] = player

    changed = True
    while changed:
        changed = False
        for y in range(h):
            for x in range(w):
                if not wall2[y+1,x+1] and \
                   ( reachable[y+1,x+2] or \
                     reachable[y+1,x  ] or \
                     reachable[y+2,x+1] or \
                     reachable[y,  x+1] ) and \
                     not reachable[y+1,x+1]:
                    reachable[y+1,x+1] = True
                    changed = True
    return reachable[1:h+1,1:w+1]


# using the reachability analysis, extract the tiles that are relevant
# for planning, i.e., reachable tiles + walls

def compute_relevant(init_layout):
    # reachability analysis
    player = (init_layout == pddlgym.rendering.sokoban.PLAYER)
    wall   = (init_layout == pddlgym.rendering.sokoban.WALL)
    print(f"{wall.sum()} wall objects:")
    print(wall)
    reachable = compute_reachability(wall,player)
    print(f"{reachable.sum()} reachable objects:")
    print(reachable)
    relevant = np.maximum(reachable, wall)
    print(f"{relevant.sum()} relevant objects:")
    print(relevant)
    relevant = relevant.reshape(-1)
    return relevant



# convert the plan returned by run_planner into a sequence of gym actions
def plan_to_actions(env,init_state,plan):
    actions = []
    for s in plan:
        a = pddlgym.parser.parse_plan_step(
                s,
                env.domain.operators.values(),
                env.action_predicates,
                init_state.objects,
                operators_as_actions=env.operators_as_actions
            )
        actions.append(a)
    return actions




# implementing validators

from keras.layers import Input, Reshape
from keras.models import Model
import keras.backend.tensorflow_backend as K
import tensorflow as tf


def setup():
    pass


archive = {"loaded":False}
default_layout_archive = archive_path("layout",float("inf"),False,False,2,False)
def load(layout_archive):
    if not archive["loaded"]:
        archive["panels"] = \
            np.array([ resize(pddlgym.rendering.sokoban.TOKEN_IMAGES[enum][:,:,:3],(tile,tile,3)) # 0-1
                       for enum in range(pddlgym.rendering.sokoban.NUM_OBJECTS) ])
        with np.load(layout_archive) as data:
            archive["pres"] = data["pres"]
            archive["sucs"] = data["sucs"]
            archive["loaded"] = True



threshold = 0.11
def build_error(s, height, width):
    objtypes = pddlgym.rendering.sokoban.NUM_OBJECTS
    s = K.reshape(s,[-1,height,tile,width,tile,3])
    s = K.permute_dimensions(s, [0,1,3,2,4,5])
    s = K.reshape(s,[-1,height,width,1,tile,tile,3])
    s = K.tile(s, [1,1,1,objtypes,1,1,1])

    allpanels = K.constant(archive["panels"])
    # [objtypes, tile, tile, 3]
    allpanels = K.reshape(allpanels, [1,1,1,objtypes,tile,tile,3])
    # [1,     1, 1, objtypes, tile, tile, 3]
    allpanels = K.tile(allpanels, [K.shape(s)[0], height, width, 1, 1, 1, 1])
    # [batch, H, W, objtypes, tile, tile, 3]

    # panel-wise errors by L-inf norm
    error = K.abs(s - allpanels)
    error = K.max(error, axis=(4,5,6))
    # [batch, H, W, objtypes]
    return error


def build(height,width,verbose):
    states = Input(shape=(height*tile,width*tile,3))
    error = build_error(states, height, width)
    matches = K.cast(K.less_equal(error, threshold), 'float32')
    # batch, h, w, panel

    layout = K.argmin(error, axis=3)
    # layout = K.expand_dims(layout, axis=3)
    # batch, h, w

    num_matches = K.sum(matches, axis=3)
    # batch, h, w
    panels_ok = K.all(K.equal(num_matches, 1), (1,2))
    panels_nomatch   = K.any(K.equal(num_matches, 0), (1,2))
    panels_ambiguous = K.any(K.greater(num_matches, 1), (1,2))

    if verbose:
        return Model(states,
                     [ wrap(states, x) for x in [panels_ok,
                                                 panels_nomatch,
                                                 panels_ambiguous,
                                                 layout]])
    else:
        return Model(states,
                     [ wrap(states, x) for x in [panels_ok,
                                                 layout]])


# validate each state.
# Approach: As a preprocessing, we exhaustively enumerate the entire states for problem 02, then dump all layouts into an archive.
# We then match the input image with each tile to recover the layout.
# Finally, look for the same layout in the archive.
def validate_states(states, verbose=True, **kwargs):
    height = states.shape[1] // tile
    width  = states.shape[2] // tile
    if verbose:
        print({"height":height,"width":width})
    load(default_layout_archive)

    model = build(height, width, verbose)
    if verbose:
        panels_ok, panels_nomatch, panels_ambiguous, layout1 = model.predict(states, **kwargs)
        print(np.count_nonzero(panels_ok),       "images have panels all of which correctly match exactly 1 panel each")
        print(np.count_nonzero(panels_nomatch),  "images have some panels which are unlike any panels")
        print(np.count_nonzero(panels_ambiguous),"images have some panels which match >2 panels")
        print("regardless of the match results, correctness is checked by the layout of the environment computed by the L2-closest panel")
    else:
        panels_ok, layout1 = model.predict(states, **kwargs)

    B, H, W = layout1.shape

    layout2 = np.concatenate((archive["pres"], archive["sucs"]))

    # layout1: [batch1, H, W]
    # layout2: [batch2, H, W, 1]

    layout2 = layout2.reshape((-1,1,H,W))
    # layout2 ->  [batch2, 1, H, W]

    # find the same states -> [batch2,batch1]
    is_same_state = np.all(layout1 == layout2, axis=(2,3))
    # -> [batch1]
    same_state_found = np.any(is_same_state,axis=0)
    return same_state_found


layout_to_string = {
    pddlgym.rendering.sokoban.CLEAR : "CLEAR",
    pddlgym.rendering.sokoban.PLAYER : "PLAYER",
    pddlgym.rendering.sokoban.STONE : "STONE",
    pddlgym.rendering.sokoban.STONE_AT_GOAL : "STONE_AT_GOAL",
    pddlgym.rendering.sokoban.GOAL : "GOAL",
    pddlgym.rendering.sokoban.WALL : "WALL"
}

# validate the transition.
# we can't use the approach used for states for transitions because the archive contains all states but not all transitions ---
# which I overlooked. (It uses dijkstra search to enumerate states, therefore there is only one transition for each state as a desitnation.)
def validate_transitions(transitions, check_states=True, verbose=True, **kwargs):
    pres = np.array(transitions[0])
    sucs = np.array(transitions[1])

    height = pres.shape[1] // tile
    width  = sucs.shape[2] // tile

    if check_states:
        valid_pres = validate_states(pres,verbose=verbose,**kwargs)
        valid_sucs = validate_states(sucs,verbose=verbose,**kwargs)

    load(default_layout_archive)

    # check_states is ignored; it is tested anyways

    model = build(height, width, False)

    _, layout_pres = model.predict(pres, **kwargs)
    _, layout_sucs = model.predict(sucs, **kwargs)

    results = np.zeros(len(pres),dtype=bool)
    for i, (pre, suc) in enumerate(zip(layout_pres,layout_sucs)):
        if check_states:
            if not valid_pres[i]:
                continue
            if not valid_sucs[i]:
                continue
        pos1 = np.where(pre == pddlgym.rendering.sokoban.PLAYER)
        pos2 = np.where(suc == pddlgym.rendering.sokoban.PLAYER)
        pos1a = np.array(pos1).flatten()
        pos2a = np.array(pos2).flatten()
        pos3a = pos2a + (pos2a-pos1a)
        pos3 = pos3a[0], pos3a[1]

        if np.sum(np.abs(pos1a-pos2a)) != 1:
            if verbose:
                print(f"player must always move by one: {pos1a} -> {pos2a}")
            continue

        example = pre[pos1]
        # note: for all conditions, we should check Successor State Axioms (SSA: everything else stays the same)
        # note2: PLAYER cell could turn out to be a GOAL after moving.
        # Thus, the visualzied domain could contain non-determinism.

        # (:action move
        #  :parameters (?p - thing ?from - location ?to - location ?dir - direction)
        #  :precondition (and (move ?dir)
        #                     (is-player ?p)
        #                     (at ?p ?from)
        #                     (clear ?to)
        #                     (move-dir ?from ?to ?dir))
        #  :effect (and (not (at ?p ?from))
        #               (not (clear ?to))
        #               (at ?p ?to)
        #               (clear ?from)))
        if pre[pos1] == pddlgym.rendering.sokoban.PLAYER and \
           (pre[pos2] == pddlgym.rendering.sokoban.CLEAR or \
            pre[pos2] == pddlgym.rendering.sokoban.GOAL ) and \
           (suc[pos1] == pddlgym.rendering.sokoban.CLEAR or \
            suc[pos1] == pddlgym.rendering.sokoban.GOAL ) and \
           suc[pos2] == pddlgym.rendering.sokoban.PLAYER:

            if verbose:
                print(f"action: move {pos1a} {pos2a}")

            ssa = (pre == suc)
            ssa[pos1] = True
            ssa[pos2] = True
            if not np.all(ssa):
                if verbose:
                    print(f"SSA did not hold.")
                continue

            results[i] = True
            continue
           
        # (:action push-to-goal
        #  :parameters (?p - thing ?s - thing ?ppos - location ?from - location ?to - location ?dir - direction)
        #  :precondition (and (move ?dir)
        #                     (is-player ?p)
        #                     (is-stone ?s)
        #                     (at ?p ?ppos)
        #                     (at ?s ?from)
        #                     (clear ?to)
        #                     (move-dir ?ppos ?from ?dir)
        #                     (move-dir ?from ?to ?dir)
        #                     (is-goal ?to))
        #  :effect (and (not (at ?p ?ppos))
        #               (not (at ?s ?from))
        #               (not (clear ?to))
        #               (at ?p ?from)
        #               (at ?s ?to)
        #               (clear ?ppos)
        #               (at-goal ?s)))
        if pre[pos1] == pddlgym.rendering.sokoban.PLAYER and \
           (pre[pos2] == pddlgym.rendering.sokoban.STONE or \
            pre[pos2] == pddlgym.rendering.sokoban.STONE_AT_GOAL) and \
           pre[pos3] == pddlgym.rendering.sokoban.GOAL and \
           (suc[pos1] == pddlgym.rendering.sokoban.CLEAR or \
            suc[pos1] == pddlgym.rendering.sokoban.GOAL ) and \
           suc[pos2] == pddlgym.rendering.sokoban.PLAYER and \
           suc[pos3] == pddlgym.rendering.sokoban.STONE_AT_GOAL:
            if verbose:
                print(f"action: push-to-goal {pos1a} {pos2a} {pos3a}")

            ssa = (pre == suc)
            ssa[pos1] = True
            ssa[pos2] = True
            ssa[pos3] = True
            if not np.all(ssa):
                if verbose:
                    print(f"SSA did not hold.")
                continue

            results[i] = True
            continue
            
        # (:action push-to-nongoal
        #  :parameters (?p - thing ?s - thing ?ppos - location ?from - location ?to - location ?dir - direction)
        #  :precondition (and (move ?dir)
        #                     (is-player ?p)
        #                     (is-stone ?s)
        #                     (at ?p ?ppos)
        #                     (at ?s ?from)
        #                     (clear ?to)
        #                     (move-dir ?ppos ?from ?dir)
        #                     (move-dir ?from ?to ?dir)
        #                     (is-nongoal ?to))
        #  :effect (and (not (at ?p ?ppos))
        #               (not (at ?s ?from))
        #               (not (clear ?to))
        #               (at ?p ?from)
        #               (at ?s ?to)
        #               (clear ?ppos)
        #               (not (at-goal ?s))))
        if pre[pos1] == pddlgym.rendering.sokoban.PLAYER and \
           (pre[pos2] == pddlgym.rendering.sokoban.STONE or \
            pre[pos2] == pddlgym.rendering.sokoban.STONE_AT_GOAL) and \
           pre[pos3] == pddlgym.rendering.sokoban.CLEAR and \
           (suc[pos1] == pddlgym.rendering.sokoban.CLEAR or \
            suc[pos1] == pddlgym.rendering.sokoban.GOAL ) and \
           suc[pos2] == pddlgym.rendering.sokoban.PLAYER and \
           suc[pos3] == pddlgym.rendering.sokoban.STONE:
            if verbose:
                print(f"action: push-to-nongoal {pos1a} {pos2a} {pos3a}")

            ssa = (pre == suc)
            ssa[pos1] = True
            ssa[pos2] = True
            ssa[pos3] = True
            if not np.all(ssa):
                if verbose:
                    print(f"SSA did not hold.")
                continue

            results[i] = True
            continue

        if verbose:
            print(f"none of the actions matched.")
            print(f" {pos1a}={layout_to_string[int(pre[pos1])]} {pos2a}={layout_to_string[int(pre[pos2])]} {pos3a}={layout_to_string[int(pre[pos3])]} ->  ")
            print(f" {pos1a}={layout_to_string[int(suc[pos1])]} {pos2a}={layout_to_string[int(suc[pos2])]} {pos3a}={layout_to_string[int(suc[pos3])]}")

    return results

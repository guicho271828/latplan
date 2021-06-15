import json
import os.path
import random
from .stacktrace import print_object
import datetime
from .util import ensure_list, NpEncoder, gpu_info

################################################################

parameters = {}

################################################################
# interrupt / exception handling

from tensorflow.python.framework.errors_impl import ResourceExhaustedError, InvalidArgumentError

import signal
class SignalInterrupt(Exception):
    """Raised when a signal handler was invoked"""
    def __init__(self,signal,frame):
        print("Received Signal",signal)
        self.signal = signal
        self.frame  = frame
        raise self

signal.signal(signal.SIGUSR2,SignalInterrupt)

class InvalidHyperparameterError(Exception):
    """Raised when the hyperparameter is not valid"""
    def __init__(self,*args,**kwargs):
        self.args = args
    def __repr__(self):
        return "InvalidHyperparameterError("+",".join(self.args)+")"
    def __str__(self):
        return "InvalidHyperparameterError("+",".join(self.args)+")"
    pass

class HyperparameterGenerationError(Exception):
    """Raised when hyperparameter generation failed """
    pass


################################################################
# save/load

import time
import math
import subprocess
def call_with_lock(path,fn):
    subprocess.call(["mkdir","-p",path])
    lock = path+"/lock"
    first = True
    while True:
        try:
            with open(lock,"x") as f:
                try:
                    result = fn()
                finally:
                    subprocess.run(["rm",lock])
            break
        except FileExistsError:
            if first:
                print("waiting for lock...")
                first = False
            time.sleep(1)
    return result

# append a new log entry into a file
def save_history(path,obj):
    print("logging the results")
    with open(os.path.join(path,"grid_search.log"), 'a') as f:
        json.dump(obj, f, cls=NpEncoder)
        f.write("\n")
    return load_history(path)

# load the past history of runs to continue the search that was previously terminated
def load_history(path):
    log = os.path.join(path,"grid_search.log")
    if os.path.exists(log):
        # print("loading previous results: ",log)
        open_list  = []
        close_list = {}
        for hist in stream_read_json(log):
            hist = ensure_list(hist[0]), *hist[1:]
            open_list.insert(0,tuple(hist))
            key = _key(hist[1])
            if key in close_list: # there could be duplicates
                close_list[key] = min(close_list[key], hist[0])
            else:
                close_list[key] = hist[0]
        open_list.sort(key=lambda x: x[0])
        return open_list, close_list
    else:
        return [], {}

# from https://stackoverflow.com/questions/6886283/how-i-can-i-lazily-read-multiple-json-values-from-a-file-stream-in-python
def stream_read_json(fn):
    start_pos = 0
    with open(fn, 'r') as f:
        while True:
            try:
                obj = json.load(f)
                yield obj
                return
            except json.JSONDecodeError as e:
                f.seek(start_pos)
                json_str = f.read(e.pos)
                if json_str == '':
                    return
                obj = json.loads(json_str)
                start_pos += e.pos
                yield obj


__dot_counter = 0
def print_dot():
    global __dot_counter
    if __dot_counter > 100:
        __dot_counter = 0
        print()
    else:
        __dot_counter += 1
    print(".",end="",flush=True)
    pass


################################################################
# algorithms

# single iteration of NN training
def nn_task(network, path, train_in, train_out, val_in, val_out, parameters, resume=False):
    print("class precedence list:")
    for c in network.mro():
        print(" ",c)
    print("clearning tf session")
    import keras.backend
    keras.backend.clear_session()
    print("cleared tf session")
    net = network(path,parameters=parameters)
    net.train(train_in,
              val_data=val_in,
              train_data_to=train_out,
              val_data_to=val_out,
              resume=resume,
              **parameters,)
    import numpy as np
    error = np.array(net.evaluate(val_in,val_out,batch_size=100,verbose=0))
    error = np.nan_to_num(error,nan=float("inf"))
    return net, error


keys_to_ignore = ["time_start","time_duration","time_end","HOSTNAME","LSB_JOBID","gpu","mean","std"]

def _key(config):
    def tuplize(x):
        if isinstance(x,list):
            return tuple([tuplize(y) for y in x])
        else:
            return x
    return tuple( tuplize(v) for k, v in sorted(config.items())
                  if k not in keys_to_ignore)


def _runs(open_list):
    return len([ tag for _, _, tag in open_list if tag is None])

def _update_best(artifact, eval, config, limit, open_list, report, report_best):
    if report:
        report(artifact,eval)
    if (_key(open_list[0][1]) == _key(config)):
        print("Found a better parameter.")
        print(f"new best = {eval}")
        if _runs(open_list) > 1:
            print(f"old best = {open_list[1][0]}")
        else:
            print(f"old best : this is a first run, so there is no old best")
        if _runs(open_list) <= limit:
            if report_best:
                report_best(artifact,eval)
        else:
            print("however, this result is not used because it is found after the limit.")
    return


def _select(list):
    return list[random.randint(0,len(list)-1)]

def _ensure_hyperparameters_list(parameters):
    return {
        k : ensure_list(v)
        for k, v in parameters.items()
    }

def _random_configs(parameters):
    while True:
        yield { k: random.choice(v) for k,v in parameters.items() }

def _all_configs(parameters):
    import itertools
    names  = [ k for k, _ in parameters.items()]
    values = [ v for _, v in parameters.items()]
    for config_values in itertools.product(*values):
        yield { k:v for k,v in zip(names,config_values) }



def grid_search(task, parameters, path,
                report=None, report_best=None,
                shuffle=True,
                limit=10000):
    parameters = _ensure_hyperparameters_list(parameters)

    open_list, close_list = call_with_lock(path,lambda : load_history(path))

    def _iter(config):
        def fn1():
            open_list, close_list = load_history(path)
            if _key(config) in close_list:
                raise HyperparameterGenerationError()
            else:
                time_start = datetime.datetime.now()
                config["time_start"]   = time_start.isoformat(timespec='milliseconds')[5:]
                # insert infinity and block the duplicated effort.
                # Third field indicating the placeholder
                save_history(path, (float("inf"), config, "placeholder"))
                return time_start
        time_start = call_with_lock(path, fn1)
        artifact, eval = task(config)
        time_end = datetime.datetime.now()
        config["time_end"]     = time_end.isoformat(timespec='milliseconds')[5:]
        config["time_duration"]    = str(time_end - time_start)
        open_list, close_list = call_with_lock(path,lambda : save_history(path, (eval, config, None)))
        _update_best(artifact, eval, config, limit, open_list, report, report_best)
        return open_list, close_list


    if shuffle:
        gen = _random_configs(parameters)
    else:
        gen = _all_configs(parameters)
    try:
        for i,config in enumerate(gen):
            if i > limit:
                break
            if _key(config) in close_list:
                continue
            try:
                open_list, close_list = _iter(config)
            except InvalidArgumentError as e:
                print(e)
            except ResourceExhaustedError as e:
                print(e)
            except InvalidHyperparameterError as e:
                print(e)
    except SignalInterrupt as e:
        print("received",e.signal,", optimization stopped")
    return


def _neighbors(parent,parameters):
    "Returns all dist-1 neighbors"
    results = []
    for k, _ in parent.items():
        if k in parameters:     # If a hyperparameter is made non-configurable, it could be missing here
            for v in parameters[k]:
                if parent[k] is not v:
                    other = parent.copy()
                    other[k] = v
                    results.append(other)
    return results

def _check_missing_hyperparameter(parent,parameters):
    """Parameter list could be updated and new jobs may be run under a new parameter list which has a new entry.
    However, the parent may hold those values in default_parameters, missing the values of its own.
    _neighbor only consider the oppsite case: when the parameters are removed.
    _crossover do not consider those values.
    As a result, the value is missing in both the new default_parameters, and the parent parameters.
    This code checks for any hyperparameters in the parents that must be configured.
"""
    for k, v in parameters.items():
        if k not in parent:
            # is missing in the parents
            parent[k] = random.choice(v)
    # for k in parent.items():
    #     if k not in parameters:
    #         # is already in the new default_parameters
    #         del parent[k]
    return parent

def _crossover(parent1,parent2):
    child = {}
    for k,v1 in parent1.items():
        v2 = parent2[k]
        if random.random() < 0.5:
            child[k] = v1
        else:
            child[k] = v2
    return child

def _adjusted_inverse_weighted_sampling(top_k):
    """Roulette-based selection where the scores are softplus of the original.
By applying a softplus, the "optimal solution" is assumed to be -inf.
Unlike previous versions, it works on negative scores and it is scale invariant.
"""
    import numpy as np
    import math
    # top_k = [ elem for elem in top_k if not math.isinf(elem[0])]
    pevals = np.array([ elem[0] for elem in top_k])

    weights = np.log(1 + np.exp(pevals))

    pos = np.where(np.cumsum(weights) >= np.random.uniform(0.0,1.0))[0][0]

    # try this to understand
    # np.cumsum([0,1,2,3])
    # np.cumsum([0,1,2,3]) < 1.5
    # np.where(np.cumsum([0,1,2,3]) < 1.5)
    # np.where(np.cumsum([0,1,2,3]) < 1.5)[0][-1]

    return top_k[pos]

def _generate_child_by_crossover(open_list, close_list, k, parameters):
    top_k = open_list[:k]
    peval1, parent1, *_ = _adjusted_inverse_weighted_sampling(top_k)
    peval2, parent2, *_ = _adjusted_inverse_weighted_sampling(top_k)
    while parent1 == parent2:
        peval2, parent2, *_ = _adjusted_inverse_weighted_sampling(top_k)

    non_mutated_child = _crossover(parent1, parent2)
    non_mutated_child = _check_missing_hyperparameter(non_mutated_child, parameters)
    children = _neighbors(non_mutated_child, parameters)
    open_children = []
    for c in children:
        if _key(c) not in close_list:
            open_children.append(c)
    if len(open_children) > 0:
        child = _select(open_children)
        print("parent1: ", parent1)
        print("peval1 : ", peval1)
        print("parent2: ", parent2)
        print("peval2 : ", peval2)
        print("child  : ", child)
        print("attempted trials : ", tried)
        return child
    else:
        raise HyperparameterGenerationError()

def simple_genetic_search(task, parameters, path,
                          initial_population=20,
                          population=10,
                          limit=float('inf'),
                          report=None, report_best=None,):
    "Initialize a queue by evaluating N nodes. Select 2 parents randomly from top N nodes and perform uniform crossover. Fall back to LGBFS on a fixed ratio (as a mutation)."
    parameters = _ensure_hyperparameters_list(parameters)

    # assert 2 <= initial_population
    # if not (2 <= initial_population):
    #     print({"initial_population":initial_population},"is superceded by",{"initial_population":2},". initial_population must be larger than equal to 2",)
    #     initial_population = 2
    
    # assert initial_population <= limit
    if not (initial_population <= limit):
        print({"initial_population":initial_population},"is superceded by",{"limit":limit},". limit must be larger than equal to the initial population",)
        initial_population = limit
    
    # assert population <= initial_population
    if not (population <= initial_population):
        print({"population":population},"is superceded by",{"initial_population":initial_population},". initial_population must be larger than equal to the population",)
        population = initial_population

    open_list, close_list = call_with_lock(path,lambda : load_history(path))

    def _iter(config):
        import os
        for key in ["HOSTNAME", "LSB_JOBID"]:
            try:
                config[key] = os.environ[key]
            except KeyError:
                pass
        try:
            config["gpu"] = gpu_info()[0]["name"]
        except:
            # not very important
            pass
        def fn1():
            open_list, close_list = load_history(path)
            if _key(config) in close_list:
                raise HyperparameterGenerationError()
            else:
                time_start = datetime.datetime.now()
                config["time_start"]   = time_start.isoformat(timespec='milliseconds')[5:]
                # insert infinity and block the duplicated effort.
                # Third field indicating the placeholder
                save_history(path, (float("inf"), config, "placeholder"))
                return time_start
        time_start = call_with_lock(path, fn1)
        artifact, eval = task(config)
        time_end = datetime.datetime.now()
        config["time_end"]     = time_end.isoformat(timespec='milliseconds')[5:]
        config["time_duration"]    = str(time_end - time_start)
        open_list, close_list = call_with_lock(path,lambda : save_history(path, (eval, config, None)))
        _update_best(artifact, eval, config, limit, open_list, report, report_best)
        return open_list, close_list


    try:
        print("Simple GA: Generating an initial population")

        max_trials = 100
        gen_config = _random_configs(parameters)
        while _runs(open_list) <= initial_population:
            done = False
            for _, config in zip(range(max_trials),gen_config):
                try:
                    open_list, close_list = _iter(config)
                    return
                except ResourceExhaustedError as e:
                    open_list, close_list = call_with_lock(path,lambda : save_history(path, (float("inf"), config, "oom")))
                    print("Simple GA: OOM!")
                    print(e)
                except InvalidHyperparameterError as e:
                    open_list, close_list = call_with_lock(path,lambda : save_history(path, (float("inf"), config, "invalid")))
                    print("Simple GA: Invalid config!")
                    print(e)
                except HyperparameterGenerationError as e:
                    print_dot()
            if not done:
                print(f"Simple GA: failed to generate a valid configuration after {max_trials} trials.")
                return

        print("Simple GA: Generated an initial population")
        while _runs(open_list) <= limit:
            done = False
            for _ in range(max_trials):
                try:
                    config = _generate_child_by_crossover(open_list, close_list, population, parameters)
                    open_list, close_list = _iter(config)
                    done = True
                    break
                except ResourceExhaustedError as e:
                    open_list, close_list = call_with_lock(path,lambda : save_history(path, (float("inf"), config, "oom")))
                    print("Simple GA: OOM!")
                    print(e)
                except InvalidHyperparameterError as e:
                    open_list, close_list = call_with_lock(path,lambda : save_history(path, (float("inf"), config, "invalid")))
                    print("Simple GA: Invalid config!")
                    print(e)
                except HyperparameterGenerationError as e:
                    print(f"Simple GA: Doubling the population size. {population} -> {population*2}")
                    population *= 2
            if not done:
                print(f"Simple GA: failed to generate a valid configuration after {max_trials} trials.")
                return
        print(f"Simple GA: generation limit reached. {_runs(open_list)} > limit:{limit}")
    except SignalInterrupt as e:
        call_with_lock(path,lambda : save_history(path, (float("inf"), config, f"signal_{e.signal}")))
        print(f"Simple GA: received signal {e.signal}, optimization stopped")
    return


# do not run it in parallel.
def reproduce(task, path, report=None, report_best=None, limit=3):

    open_list, close_list = load_history(path)

    reproduce_list = []

    print("Reproducing the best results from the logs")
    try:
        for i in range(limit):
            config = open_list[0][1].copy()
            config["reproduce_trial"] = i
            artifact, eval = task(config)
            reproduce_list.append((eval, config, None))
            reproduce_list.sort(key=lambda x: x[0])
            _update_best(artifact, eval, config, limit, reproduce_list, report, report_best)

    except SignalInterrupt as e:
        print("received",e.signal,", optimization stopped")
    return


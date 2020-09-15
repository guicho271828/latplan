import json
import os.path
from timeout_decorator import timeout, TimeoutError
import random
from .stacktrace import print_object

from tensorflow.python.framework.errors_impl import ResourceExhaustedError

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
    pass

class HyperparameterGenerationError(Exception):
    """Raised when the hyperparameter generation failed """
    pass

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

# append the new log entry into a file
def save_history(path,obj):
    print("logging the results")
    with open(os.path.join(path,"grid_search.log"), 'a') as f:
        json.dump(obj, f)
        f.write("\n")
    return load_history(path)

# load the past history of runs to continue the search that was previously terminated
def load_history(path):
    log = os.path.join(path,"grid_search.log")
    if os.path.exists(log):
        print("loading the previous results: ",log)
        open_list  = []
        close_list = {}
        for hist in stream_read_json(log):
            # print_object({"loaded":hist})
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
                obj = json.loads(json_str)
                start_pos += e.pos
                yield obj

def save_default_parameters(path,default_parameters):
    fname = os.path.join(path,"default_parameters.json")
    def fn():
        if not os.path.exists(fname):
            with open(fname,"w") as f:
                json.dump(default_parameters, f)
    call_with_lock(path,fn)
    return

def load_default_parameters(path):
    fname = os.path.join(path,"default_parameters.json")
    with open(fname) as f:
        return json.load(f)

# single iteration of NN training
def nn_task(network, path, train_in, train_out, val_in, val_out, parameters):
    net = network(path,parameters=parameters)
    net.train(train_in,
              val_data=val_in,
              train_data_to=train_out,
              val_data_to=val_out,
              save=False,
              **parameters,)
    error = net.evaluate(val_in,val_out,batch_size=100,verbose=0)
    if math.isnan(error):
        error = float("inf")
    return net, error

def merge_hash(a, b):
    c = a.copy()
    for key, value in b.items():
        c[key] = value
    return c

def _select(list):
    return list[random.randint(0,len(list)-1)]

def _update_best(artifact, eval, config, best, report, report_best):
    if report and (artifact is not None):
        report(artifact)
    print("Evaluation result for:\n{}\neval = {}".format(config,eval))
    if best['eval'] is None or eval < best['eval']:
        print("Found a better parameter:\n{}\neval:{} old-best:{}".format(config,eval,best['eval']))
        if report_best and (artifact is not None):
            report_best(artifact)
        best['params'] = config
        best['eval'] = eval
        best['artifact'] = artifact
    else:
        del artifact

def _random_configs(parameters):
    while True:
        yield { k: random.choice(v) for k,v in parameters.items() }

def _all_configs(parameters):
    import itertools
    names  = [ k for k, _ in parameters.items()]
    values = [ v for _, v in parameters.items()]
    for config_values in itertools.product(*values):
        yield { k:v for k,v in zip(names,config_values) }

def _final_report(best):
    from colors import bold
    print(bold("*** Best parameter: ***\n{}\neval: {}".format(best['params'],best['eval'])))
    return

def grid_search(task, default_config, parameters, path,
                report=None, report_best=None,
                shuffle=True,
                limit=10000):
    best = {'eval'    :None, 'params'  :None, 'artifact':None}
    save_default_parameters(path, default_config)
    open_list, close_list = call_with_lock(path,lambda : load_history(path))
    if len(open_list) > 0:
        _update_best(None, open_list[0][0], open_list[0][1], best, None, None)

    def _iter(config):
        nonlocal open_list, close_list
        parameters = merge_hash(default_config,config)
        parameters["current_best"] = best["eval"]
        artifact, eval = task(parameters)
        open_list, close_list = call_with_lock(path,lambda :  save_history(path, (eval, config, default_config)))
        _update_best(artifact, eval, config, best, report, report_best)

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
                _iter(config)
            except ResourceExhaustedError as e:
                print(e)
            except InvalidHyperparameterError as e:
                print(e)
    except SignalInterrupt as e:
        print("received",e.signal,", optimization stopped")
    finally:
        _final_report(best)
    return best['artifact'],best['params'],best['eval']

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
    """Parameter list could be updated and new jobs may be run under the new parameter list which has a new entry.
    However, the parent may hold those values in the default_parameters, missing the values of its own.
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



def _key(config):
    def tuplize(x):
        if isinstance(x,list):
            return tuple(x)
        else:
            return x
    return tuple( tuplize(v) for _, v in sorted(config.items()))

def _crossover(parent1,parent2):
    child = {}
    for k,v1 in parent1.items():
        v2 = parent2[k]
        if random.random() < 0.5:
            child[k] = v1
        else:
            child[k] = v2
    return child

def _inverse_weighted_select(lst):
    weights = [ 1/entry[0] for entry in lst ]
    cum_weights = []
    cum = 0.0
    for w in weights:
        cum += w
        cum_weights.append(cum)

    pivot = random.random() * cum
    selected = len(cum_weights)-1
    for i, th in enumerate(cum_weights):
        if pivot < th:
            selected = i
            break

    return lst[selected]

def _generate_child_by_crossover(open_list, close_list, k, max_trial, parameters):
    top_k = open_list[:k]
    for tried in range(max_trial):
        peval1, parent1, *_ = _inverse_weighted_select(top_k)
        peval2, parent2, *_ = _inverse_weighted_select(top_k)
        while parent1 == parent2:
            peval2, parent2, *_ = _inverse_weighted_select(top_k)

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
    print("Simple GA: crossover failed after {} trials".format(max_trial))
    raise HyperparameterGenerationError()

def simple_genetic_search(task, default_config, parameters, path,
                          initial_population=20,
                          population=10,
                          limit=float('inf'),
                          report=None, report_best=None,):
    "Initialize the queue by evaluating the N nodes. Select 2 parents randomly from top N nodes and perform the uniform crossover. Fall back to LGBFS on a fixed ratio (as a mutation)."
    best = {'eval'    :None, 'params'  :None, 'artifact':None}

    save_default_parameters(path, default_config)
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

    # Python sucks! It does not even have a normal lambda.
    def fn():
        open_list, close_list = load_history(path)
        if len(open_list) > 0:
            _update_best(None, open_list[0][0], open_list[0][1], best, None, None)
        return open_list, close_list
    open_list, close_list = call_with_lock(path, fn)

    def _iter(config):
        def fn1():
            open_list, close_list = load_history(path)
            if _key(config) in close_list:
                raise HyperparameterGenerationError()
            else:
                # insert infinity and block the duplicated effort.
                # Third field indicating the placeholder
                save_history(path, (float("inf"), config, "placeholder"))
        call_with_lock(path, fn1)
        parameters = merge_hash(default_config,config)
        parameters["current_best"] = best["eval"]
        artifact, eval = task(parameters)
        def fn2():
            open_list, close_list = save_history(path, (eval, config, None))
            import math
            if (open_list[0][1] == config) and (len([ tag for _,_,tag in open_list if tag != "placeholder"]) <= limit):
                _update_best(artifact, eval, config, best, report, report_best)
            return open_list, close_list
        return call_with_lock(path, fn2)

    try:
        print("Simple GA: Generating the initial population")

        gen_config = _random_configs(parameters)
        try:
            while len([ tag for _, _, tag in open_list if tag is None]) < initial_population:
                for config in gen_config:
                    try:
                        open_list, close_list = _iter(config)
                        break
                    except ResourceExhaustedError as e:
                        print("OOM!")
                        print(e)
                    except InvalidHyperparameterError as e:
                        print("invalid config!")
                        pass
                    except HyperparameterGenerationError as e:
                        print("duplicate config!")
                        pass
        except StopIteration:   # from gen_config
            pass

        print("Simple GA: Generated the initial population")
        while len(open_list) < limit:
            done = False
            while not done:
                try:
                    child = _generate_child_by_crossover(open_list, close_list, population, 10000, parameters)
                    done = True
                except HyperparameterGenerationError as e:
                    print(e)
                    print("Simple GA: Increasing populations {} -> {}".format(population,population*2))
                    population = population*2
                    if population > len(close_list.items()):
                        print("Simple GA: Search space exhausted.")
                        return best['artifact'],best['params'],best['eval']
            
            try:
                open_list, close_list = _iter(child)
            except ResourceExhaustedError as e:
                print(e)
            except InvalidHyperparameterError as e:
                pass
    except SignalInterrupt as e:
        print("received",e.signal,", optimization stopped")
    finally:
        _final_report(best)
    return best['artifact'],best['params'],best['eval']


# do not run it in parallel.
def reproduce(task, path, report=None, report_best=None, limit=3):
    best = {'eval'    :None, 'params'  :None, 'artifact':None}

    open_list, close_list = load_history(path)
    default_config = load_default_parameters(path)

    def _iter(config):
        parameters = merge_hash(default_config,config)
        parameters["current_best"] = best["eval"]
        artifact, eval = task(parameters)
        _update_best(artifact, eval, config, best, report, report_best)

    print("Reproducing the best results from the log")
    try:
        for _ in range(limit):
            _iter(open_list[0][1])
    except SignalInterrupt as e:
        print("received",e.signal,", optimization stopped")
    finally:
        _final_report(best)
    return best['artifact'],best['params'],best['eval']

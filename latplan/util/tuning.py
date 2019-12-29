import os.path
from timeout_decorator import timeout, TimeoutError
import random

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

# append the new log entry into a file
def save_history(path,obj):
    print("logging the results")
    with open(os.path.join(path,"grid_search.log"), 'a') as f:
        import json
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
            print("loaded:",hist)
            open_list.insert(0,tuple(hist))
            close_list[_key(hist[1])] = hist[0]
        open_list.sort(key=lambda x: x[0])
        return open_list, close_list
    else:
        return [], {}

# from https://stackoverflow.com/questions/6886283/how-i-can-i-lazily-read-multiple-json-values-from-a-file-stream-in-python
def stream_read_json(fn):
    import json
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

# single iteration of NN training
def nn_task(network, path, train_in, train_out, val_in, val_out, parameters):
    net = network(path,parameters=parameters)
    net.train(train_in,
              val_data=val_in,
              train_data_to=train_out,
              val_data_to=val_out,
              save=False,
              **parameters,)
    error = net.net.evaluate(val_in,val_out,batch_size=100,verbose=0)
    return net, error

def merge_hash(a, b):
    c = a.copy()
    for key, value in b.items():
        c[key] = value
    return c

def _select(list):
    return list[random.randint(0,len(list)-1)]

def _update_best(artifact, eval, config, results, best, report, report_best):
    results.append((eval, config))
    if report:
        report(artifact)
    print("Evaluation result for:\n{}\neval = {}".format(config,eval))
    print("Current results:")
    results.sort(key=lambda result: result[0])
    [ print(r) for r in results]
    if best['eval'] is None or eval < best['eval']:
        print("Found a better parameter:\n{}\neval:{} old-best:{}".format(config,eval,best['eval']))
        if report_best:
            report_best(artifact)
        del best['artifact']
        best['params'] = config
        best['eval'] = eval
        best['artifact'] = artifact
    else:
        del artifact

def _random_configs(parameters):
    while True:
        yield { k: random.choice(v) for k,v in parameters.items() }

def _final_report(best,results):
    from colors import bold
    print(bold("*** Best parameter: ***\n{}\neval: {}".format(best['params'],best['eval'])))
    print(results)
    return

def _neighbors(parent,parameters):
    "Returns all dist-1 neighbors"
    results = []
    for k, _ in parent.items():
        if k in parameters:     # HACK! HACK! remove in the next run. 2019/12/29
            for v in parameters[k]:
                if parent[k] is not v:
                    other = parent.copy()
                    other[k] = v
                    results.append(other)
    return results

def _key(config):
    return tuple( v for _, v in sorted(config.items()))

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
        peval1, parent1, _ = _inverse_weighted_select(top_k)
        peval2, parent2, _ = _inverse_weighted_select(top_k)
        while parent1 == parent2:
            peval2, parent2, _ = _inverse_weighted_select(top_k)

        child = _crossover(parent1, parent2)
        if _key(child) not in close_list:
            print("parent1: ", parent1)
            print("peval1 : ", peval1)
            print("parent2: ", parent2)
            print("peval2 : ", peval2)
            print("child  : ", child)
            print("attempted trials : ", tried)
            return child
    print("Simple GA: crossover failed after {} trials".format(max_trial))
    print("Simple GA: falling back to mutation")
    return _generate_child_by_mutation(open_list, close_list, k, max_trial, parameters)

def _generate_child_by_mutation(open_list, close_list, k, max_trial, parameters):
    top_k = open_list[:k]
    for tried in range(max_trial):
        peval, parent, _ = _inverse_weighted_select(top_k)
        children = _neighbors(parent, parameters)
        open_children = []
        for c in children:
            if _key(c) not in close_list:
                open_children.append(c)
        if len(open_children) > 0:
            child = _select(open_children)
            print("parent: ", parent)
            print("peval : ", peval)
            print("child : ", child)
            print("attempted trials : ", tried)
            return child
    print("Simple GA: mutation failed after {} trials".format(max_trial))
    print("Simple GA: Reached a local minima")
    raise HyperparameterGenerationError()

def simple_genetic_search(task, default_config, parameters, path,
                          initial_population=20,
                          population=10,
                          limit=float('inf'),
                          report=None, report_best=None,):
    "Initialize the queue by evaluating the N nodes. Select 2 parents randomly from top N nodes and perform the uniform crossover. Fall back to LGBFS on a fixed ratio (as a mutation)."
    best = {'eval'    :None, 'params'  :None, 'artifact':None}
    results       = []

    # assert 2 <= initial_population
    if not (2 <= initial_population):
        print({"initial_population":initial_population},"is superceded by",{"initial_population":2},". initial_population must be larger than equal to 2",)
        initial_population = 2
    
    # assert initial_population <= limit
    if not (initial_population <= limit):
        print({"initial_population":initial_population},"is superceded by",{"limit":limit},". limit must be larger than equal to the initial population",)
        initial_population = limit
    
    # assert population <= initial_population
    if not (population <= initial_population):
        print({"population":population},"is superceded by",{"initial_population":initial_population},". initial_population must be larger than equal to the population",)
        population = initial_population

    open_list, close_list = load_history(path)
    if len(open_list) > 0:
        _update_best(None, open_list[0][0], open_list[0][1], results, best, None, None)

    def _iter(config):
        nonlocal open_list, close_list
        artifact, eval = task(merge_hash(default_config,config))
        open_list, close_list = save_history(path, (eval, config, default_config))
        _update_best(artifact, eval, config, results, best, report, report_best)

    try:
        print("Simple GA: Generating the initial population")
        
        try:
            gen_i      = iter(range(len(open_list),initial_population))
            gen_config = _random_configs(parameters)
            
            i      = next(gen_i)
            config = next(gen_config)
            while True:
                try:
                    _iter(config)
                    i = next(gen_i)
                except InvalidHyperparameterError as e:
                    pass
                config = next(gen_config)
        except StopIteration:
            pass
        i = len(open_list)
        print("Simple GA: Generated the initial population")

        while i < limit:
            mutation_ratio = open_list[0][0] / open_list[population-1][0]
            assert mutation_ratio < 1
            print("Simple GA: best",open_list[0][0],
                  "worst",open_list[population-1][0],
                  "current mutation ratio",mutation_ratio)
            done = False
            while not done:
                try:
                    if random.random() < mutation_ratio:
                        print("Simple GA: mutation was selected")
                        child = _generate_child_by_mutation(open_list, close_list, population, 10000, parameters)
                    else:
                        print("Simple GA: crossover was selected")
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
                _iter(child)
                i += 1
            except InvalidHyperparameterError as e:
                pass
    except SignalInterrupt as e:
        print("received",e.signal,", optimization stopped")
    finally:
        _final_report(best,results)
    return best['artifact'],best['params'],best['eval']

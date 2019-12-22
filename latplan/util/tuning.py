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

def nn_task(network, path, train_in, train_out, val_in, val_out, parameters):
    net = network(path,parameters=parameters)
    net.train(train_in,
              val_data=val_in,
              train_data_to=train_out,
              val_data_to=val_out,
              save=False,
              **parameters,)
    error = net.net.evaluate(val_in,val_out,batch_size=100,verbose=0)
    print("logging the results")
    with open(net.local("grid_search.log"), 'a') as f:
        import json
        json.dump((error, parameters), f)
        f.write("\n")
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

def _random_configs(parameters,shuffle):
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

def _top_k(open_list, k):
    top_k = []
    for i in range(k):
        if open_list.empty():
            raise Exception("open list exhausted!")
        top_k.append(open_list.get())
    for parent in top_k:
        open_list.put(parent)
    return top_k

def _inverse_weighted_select(lst):
    weights = [ 1/peval for peval, _ in lst ]
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
    top_k = _top_k(open_list,k)
    for tried in range(max_trial):
        peval1, parent1 = _inverse_weighted_select(top_k)
        peval2, parent2 = _inverse_weighted_select(top_k)
        while parent1 == parent2:
            peval2, parent2 = _inverse_weighted_select(top_k)

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
    top_k = _top_k(open_list,k)
    for tried in range(max_trial):
        peval, parent = _inverse_weighted_select(top_k)
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

def simple_genetic_search(task, default_config, parameters,
                          initial_population=20,
                          population=10,
                          mutation_ratio=0.3,
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

    import queue
    open_list  = queue.PriorityQueue()
    close_list = {}

    def _iter(config):
        artifact, eval = task(merge_hash(default_config,config))
        _update_best(artifact, eval, config, results, best, report, report_best)
        close_list[_key(config)] = eval # tuples are hashable
        open_list.put((eval, config))

    try:
        print("Simple GA: Generating the initial population")
        
        try:
            gen_i      = iter(range(initial_population))
            gen_config = _random_configs(parameters, True)
            
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
        i = initial_population
        print("Simple GA: Generated the initial population")
        while i < limit:
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

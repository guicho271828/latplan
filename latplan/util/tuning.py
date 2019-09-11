from timeout_decorator import timeout, TimeoutError
import numpy.random as random

def nn_task(network, path, train_in, train_out, test_in, test_out, parameters):
    net = network(path,parameters=parameters)
    try:
        net.train(train_in,
                  test_data=test_in,
                  train_data_to=train_out,
                  test_data_to=test_out,
                  report=False,
                  **parameters,)
        error = net.net.evaluate(test_in,test_out,batch_size=100,verbose=0)
    finally:
        try:
            print("logging the results")
            with open(net.local("grid_search.log"), 'a') as f:
                import json
                json.dump((error, parameters), f)
                f.write("\n")
        except TypeError:
            pass
    return net, error

def merge_hash(a, b):
    c = a.copy()
    for key, value in b.items():
        c[key] = value
    return c

def _select(list):
    import random
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
    import itertools
    names  = [ k for k, _ in parameters.items()]
    values = [ v for _, v in parameters.items()]
    all_config_values = list(itertools.product(*values))
    if shuffle:
        random.shuffle(all_config_values)
    for config_values in all_config_values:
        yield { k:v for k,v in zip(names,config_values) }

def _final_report(best,results):
    from colors import bold
    print(bold("*** Best parameter: ***\n{}\neval: {}".format(best['params'],best['eval'])))
    print(results)
    return

def grid_search(task, default_config, parameters,
                report=None, report_best=None,
                shuffle=True,
                limit=float('inf')):
    best = {'eval'    :None, 'params'  :None, 'artifact':None}
    results       = []
    all_configs = list(_random_configs(parameters, shuffle))
    list(map(print, all_configs))
    try:
        for i,config in enumerate(all_configs):
            if i > limit:
                break
            print("{}/{} {}".format(i, len(all_configs), config))
            artifact, eval = task(merge_hash(default_config,config))
            _update_best(artifact, eval, config, results, best, report, report_best)
    finally:
        _final_report(best,results)
    return best['artifact'],best['params'],best['eval']

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

def lazy_greedy_best_first_search(task, default_config, parameters,
                                  initial_population=5,
                                  limit=float('inf'),
                                  report=None, report_best=None,):
    "Initialize the queue by evaluating the fixed number of nodes. Expand the best node based on the parent's evaluation. Randomly select, evaluate, queue a child. The parent is requeued until it expands all children."
    best = {'eval'    :None, 'params'  :None, 'artifact':None}
    results       = []
    list(map(print, _random_configs(parameters, False)))

    import queue
    open_list  = queue.PriorityQueue()
    close_list = {}

    def _iter(config):
        artifact, eval = task(merge_hash(default_config,config))
        _update_best(artifact, eval, config, results, best, report, report_best)
        close_list[_key(config)] = eval # tuples are hashable
        open_list.put((eval, config))
        
    try:
        for i,config in zip(range(initial_population),_random_configs(parameters, True)):
            _iter(config)
        i = initial_population
        while i < limit:
            i += 1
            if open_list.empty():
                break
            peval, parent = open_list.get()
            children = _neighbors(parent, parameters)

            open_children = []
            for c in children:
                if _key(c) not in close_list:
                    open_children.append(c)
            
            if len(open_children) > 0:
                open_list.put((peval, parent))
                _iter(_select(open_children))
    finally:
        _final_report(best,results)
    return best['artifact'],best['params'],best['eval']


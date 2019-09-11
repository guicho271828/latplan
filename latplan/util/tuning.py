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


def greedy_search(task, default_parameters, parameters,
                  initial_population=3, limit=10,
                  report=None, report_best=None,):
    import itertools
    names  = [ k for k, _ in parameters.items()]
    values = [ v for _, v in parameters.items()]
    all_params = list(itertools.product(*values))
    random.shuffle(all_params)
    [ print(r) for r in all_params]
    best = {'eval'    :None, 'params'  :None, 'artifact':None}

    import queue
    open_list  = queue.PriorityQueue()
    close_list = {}
    results       = []          # for displaying

    def random_configs():
        import random
        while True:
            yield { k : _select(v) for k, v in parameters.items() }

    def neighbors(parent):
        import random
        results = []
        for k in names:
            for kv in parameters[k]:
                if parent[k] is not kv:
                    other = parent.copy()
                    other[k] = kv
                    results.append(other)
        return results

    def _key(local_parameters):
        return tuple( local_parameters[k] for k in names )
    
    def _iter(local_parameters):
        artifact, eval = task(merge_hash(default_parameters,local_parameters))
        _update_best(artifact, eval, local_parameters, results, best, report, report_best)
        # 
        close_list[_key(local_parameters)] = eval # tuples are hashable
        open_list.put((eval, local_parameters))
        
    try:
        for i,local_parameters in zip(range(initial_population),random_configs()):
            _iter(local_parameters)
        for i in range(initial_population, limit):
            if open_list.empty():
                break
            _, parent = open_list.get()
            children = neighbors(parent)

            open_children = []
            for c in children:
                if _key(c) not in close_list:
                    open_children.append(c)
            
            _iter(_select(open_children))
    finally:
        print("Best parameter:\n{}\neval: {}".format(best['params'],best['eval']))
        print(results)
    return best['artifact'],best['params'],best['eval']

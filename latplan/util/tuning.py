from timeout_decorator import timeout, TimeoutError
import numpy.random as random

def merge_hash(a, b):
    c = a.copy()
    for key, value in b.items():
        c[key] = value
    return c

def grid_search(task, default_parameters, parameters,
                report=None,
                limit=float('inf')):
    best_eval     = float('inf')
    best_params   = None
    best_artifact = None
    results       = []
    import itertools
    names  = [ k for k, _ in parameters.items()]
    values = [ v for _, v in parameters.items()]
    all_params = list(itertools.product(*values))
    random.shuffle(all_params)
    [ print(r) for r in all_params]
    try:
        for i,params in enumerate(all_params):
            if i > limit:
                break
            local_parameters = { k:v for k,v in zip(names,params) }
            print("{}/{} {}".format(i, len(all_params), local_parameters))
            artifact, eval = task(merge_hash(default_parameters,local_parameters))
            results.append((eval, local_parameters))
            print("Evaluation result for:\n{}\neval = {}".format(local_parameters,eval))
            print("Current results:")
            results.sort(key=lambda result: result[0])
            [ print(r) for r in results]
            if eval < best_eval:
                print("Found a better parameter:\n{}\neval:{} old-best:{}".format(local_parameters,eval,best_eval))
                if report:
                    report(artifact)
                del best_artifact
                best_params = local_parameters
                best_eval = eval
                best_artifact = artifact
            else:
                del artifact
    finally:
        print("Best parameter:\n{}\neval: {}".format(best_params,best_eval))
        print(results)
    return best_artifact,best_params,best_eval

def best_first_search(task, default_parameters, parameters,
                      initial_population=5,
                      report=None,
                      limit=float('inf'), runtime=float('inf')):
    best = {'eval'    :float('inf'),
            'params'  :None,
            'artifact':None}
    results       = []
    results_hash  = {}
    close_list    = {}
    import itertools
    names  = [ k for k, _ in parameters.items()]
    values = [ v for _, v in parameters.items()]
    all_params = list(itertools.product(*values))
    random.shuffle(all_params)
    [ print(r) for r in all_params]
    def _iter(params):
        local_parameters = { k:v for k,v in zip(names,params) }
        print("{}/{} {}".format(i, len(all_params), local_parameters))
        artifact, eval = task(merge_hash(default_parameters,local_parameters))
        results.append((eval, local_parameters, params))
        results.sort(key=lambda result: result[0])
        results_hash[tuple(params)] = eval
        close_list[tuple(params)] = False
        print("Evaluation result for:\n{}\neval = {}".format(local_parameters,eval))
        print("Current results:")
        [ print(r) for r in results]
        if eval < best_eval:
            print("Found a better parameter:\n{}\neval:{} old-best:{}".format(local_parameters,eval,best_eval))
            if report:
                report(artifact)
            del best_artifact
            best['params'] = local_parameters
            best['eval'] = eval
            best['artifact'] = artifact
        else:
            del artifact
    try:
        for i,params in enumerate(all_params):
            if i > initial_population:
                break
            _iter(params)
        i = initial_population
        while True:
            neighbors = []
            for _, _, params in results:
                if close[params] is True:
                    continue
                close[params] = True
                for dim, current in enumerate(params):
                    for other_param in parameters[names[dim]]:
                        if other_param != current:
                            candidate = list(params).copy()
                            candidate[dim] = other_param
                            candidate = tuple(candidate)
                            if candidate in close:
                                if close[candidate] is False:
                                    neighbors.append(candidate)
                            else:
                                neighbors.append(candidate)
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if i > limit:
                    break
                i += 1
                _iter(neighbor)
    finally:
        print("Best parameter:\n{}\neval: {}".format(best_params,best_eval))
        print(results)
    return best_artifact,best_params,best_eval

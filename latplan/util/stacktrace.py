#!/usr/bin/env python3

import os.path
import sys, traceback, types, linecache
import numpy as np

def print_object(o,include_private=False,threshold=3):
    maxlinelen=1000
    maxlen=20
    def get(key):
        if isinstance(o, dict):
            return o[key]
        else:
            return getattr(o, key)
    def include(key):
        return (include_private or ("__" not in key))     \
          and not isinstance(get(key),types.FunctionType) \
          and not isinstance(get(key),types.ModuleType)   \
          and not isinstance(get(key),type)
    def remove_array(thing):
        if isinstance(thing,np.ndarray):
            return "<numpy.ndarray {:8s} {}>".format(str(thing.dtype),thing.shape)
        else:
            return thing

    def printer(thing):
        if isinstance(thing,list):
            if len(thing) > threshold:
                return [printer(remove_array(o)) for o, _ in [*list(zip(thing, range(threshold))),(f"...<{len(thing)-threshold} more>",None)]]
            else:
                return [printer(remove_array(o)) for o in thing]
        elif isinstance(thing,tuple):
            if len(thing) > threshold:
                return tuple([printer(remove_array(o)) for o in [*list(zip(thing, range(threshold))),(f"...<{len(thing)-threshold} more>",None)]])
            else:
                return tuple([printer(remove_array(o)) for o in thing])
        elif isinstance(thing,dict):
            return {k:printer(remove_array(v)) for k,v in thing.items()}
        elif isinstance(thing,str):
            return thing[:500]
        elif isinstance(thing,bytes):
            return thing[:500]
        else:
            return remove_array(thing)

    try:
        zip(o)
    except TypeError:
        print(o)
        return
    for key in o:
        try:
            if include(key):
                maxlen = max(maxlen,len(key))
        except:
            pass
    for key in o:
        try:
            if include(key):
                print("{} = {}".format(key.rjust(maxlen+4),repr(printer(get(key))))[:maxlinelen],file=sys.stderr)
        except Exception as e:
            print("{} = Error printing object : {}".format(str(key).rjust(maxlen),e),file=sys.stderr)

def format(exit=True,threshold=3):
    np.set_printoptions(threshold=25,formatter=None)
    print("Fancy Traceback (most recent call last):",file=sys.stderr)
    type, value, tb = sys.exc_info()
    
    for f, f_lineno in traceback.walk_tb(tb):
        co = f.f_code
        f_filename = co.co_filename
        f_name = co.co_name
        linecache.lazycache(f_filename, f.f_globals)
        f_locals = f.f_locals
        f_line = linecache.getline(f_filename, f_lineno).strip()

        print(" ","File",os.path.relpath(f_filename),"line",f_lineno,"function",f_name,":",f_line,file=sys.stderr)
        print_object(f_locals,threshold=threshold)
        print(file=sys.stderr)
        
    
    print(file=sys.stderr)
    print(*(traceback.format_exception_only(type,value)),file=sys.stderr)
    if exit:
        sys.exit(1)

def fn1():
    a = 1
    b = 0
    fn2(a,b)

def fn2(a,b):
    return a/b

if __name__ == '__main__':
    try:
        fn1()
    except Exception:
        format()
        print("## standard stack trace ########################################",file=sys.stderr)
        traceback.print_exc()

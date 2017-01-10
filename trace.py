import sys
trace_level = 1
def trace(fn):
    print("Tracing {}".format(fn))
    def modified(*args,**kwargs):
        global trace_level
        print("{}{}{}{}".format(("  "*trace_level),fn.__name__,args,kwargs))
        trace_level = trace_level+1
        try:
            res = fn(*args,**kwargs)
            print("{}{}{}{} returns {}".format(("  "*(trace_level-1)),fn.__name__,args,kwargs,res))
        finally:
            trace_level = trace_level-1
        return res
    return modified

def myfn3(a,b=2):
    print(a+b)

if __name__ == '__main__':
    def myfn(a,b=2):
        print(a+b)
    def myfn2(a,b=2):
        myfn(a,b)
        print(a+b)
    myfn2(1)
    myfn2(1,3)
    myfn2(1,b=3)
    myfn2 = trace(myfn2)
    myfn = trace(myfn)
    myfn2(1)
    myfn2(1,3)
    myfn2(1,b=3)
    
    myfn3(1,b=3)
    myfn3 = trace(myfn3)
    myfn3(1,b=3)
    
    class myclass:
        def __init__(self,c):
            self.c = c
        def mymethod(self,a,b=2):
            return a+b+self.c
    print(myclass)
    myclass.mymethod = trace(myclass.mymethod)
    myclass(3).mymethod(1)
    myclass(3).mymethod(1,5)
    
    

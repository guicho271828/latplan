
import time

class Timer:
    def __init__(self,message="",verbose=True):
        if message:
            print(message)
        self.verbose = verbose
    
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.verbose:
            print("Done!",self.interval,"sec")



class binary_search:
    def __init__(self,min, max):
        self.min = min
        self.max = max
        self.left = min
        self.right = max
        self.value = (min+max)/2
    def goleft(self):
        self.right = self.value
        self.value = (self.left+self.value)/2
        return self.value
    def goright(self):
        self.left = self.value
        self.value = (self.right+self.value)/2
        return self.value
    

class Class:

    def __init__(self, features):
        # features is a table containing 2 binary elements
        self.features = features

    def get_class(self):
        # the class is 0,1,2 or 3 = 2*i+j where features = [i,j] 
        # in practice: i = 0(young) or 1(old), j = 0(woman) or 1(man)
        # so that even classes are female, odd males
        # and small values (0 or 1) young, big (2 or 3) old
        return 2*self.features[0]+self.features[1]
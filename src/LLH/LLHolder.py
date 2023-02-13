
from src.LLH.LLHSet1 import LLHSet1
from src.LLH.LLHSet2 import LLHSet2
from src.LLH.LLHSet3 import LLHSet3
from src.LLH.LLHSet4 import LLHSet4


class LLHolder:
    def __init__(self, llh_set):
        if llh_set == 1:
            self.set = LLHSet1()
        elif llh_set == 2:
            self.set = LLHSet2()
        elif llh_set == 3:
            self.set = LLHSet3()
        elif llh_set == 4:
            self.set = LLHSet4()








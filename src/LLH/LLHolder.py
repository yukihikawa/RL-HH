
from src.LLH.LLHSet1 import LLHSet1
from src.LLH.LLHSet2 import LLHSet2
from src.LLH.LLHSet2_2 import LLHSet7
from src.LLH.LLHSet3 import LLHSet3
from src.LLH.LLHSet4 import LLHSet4
from src.LLH.LLHSet5 import LLHSet5
from src.LLH.LLHSet6 import LLHSet6


class LLHolder:
    def __init__(self, llh_set):
        if llh_set == 1: #原始版
            self.set = LLHSet1()
        elif llh_set == 2: # 遗传算法版
            self.set = LLHSet2()
        elif llh_set == 3:
            self.set = LLHSet3()
        elif llh_set == 4: #1的禁忌搜索版
            self.set = LLHSet4()
        elif llh_set == 5: #精简版
            self.set = LLHSet5()
        elif llh_set == 6: # 4 修改3号方法，实现机器码并行搜索
            self.set = LLHSet6()
        elif llh_set == 7: # 改进的
            self.set = LLHSet7()








import random

from src.LLH.LLHUtils import timeTaken, getMachineIdx, changeMsRandom
from src.utils import encoding
from src.utils.parser import parse

class LLHSetILS:
    def __init__(self, train = False):
        self.train = train

        # 邻域
        self.neighbourhoods = []

        # 方法
        self.local_search = []
        self.shakes = []

    def reset(self, problem_path):
        self.parameters = parse(problem_path)

        if self.train:
            self.best_solution = self.previous_solution = encoding.initializeFixedResult(self.parameters)
        else:
            self.best_solution = self.previous_solution = encoding.initializeResult(self.parameters)
        self.previous_time = self.best_time = timeTaken(self.previous_solution, self.parameters)

    def refresh_best_solution(self):
        if self.best_time > self.previous_time:
            self.best_solution = self.previous_solution
            self.best_time = self.previous_time



    # ==============================局部搜索方法==============================





    # ==============================邻域扰动方法==============================

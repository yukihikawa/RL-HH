import threading
import time
import random
from concurrent.futures import wait, ALL_COMPLETED

from src.LLH.LLHUtils import timeTaken


class apply:
    def __init__(self, solution, parameters, pool, LLH):
        self.solution = solution
        self.parameters = parameters
        self.pool = pool
        self.LLH = LLH
        self.new_solution = []
        self.new_time = []

    def applyChromTabu(self, chrom):
        """Apply chromosome to solution."""
        (os, ms) = self.solution
        # 声明一个长度和 LLH 相同的禁忌列表
        taboo = [0 for i in range(len(self.LLH))]
        prev_solution = best_solution = (os.copy(), ms.copy())
        prev_time = b_time = timeTaken(best_solution, self.parameters)
        for i in chrom:
            if sum(taboo) == len(self.LLH):
                taboo = [0 for i in range(len(self.LLH))]
            if taboo[i] == 1:
                continue
            cand_solution = self.LLH[i](prev_solution, self.parameters)
            cand_time = timeTaken(cand_solution, self.parameters)
            if cand_time < prev_time:
                prev_solution = cand_solution
                prev_time = cand_time
                taboo = [0 for i in range(len(self.LLH))]
                if cand_time < b_time:
                    best_solution = cand_solution
                    b_time = cand_time
                    # return best_solution, b_time
            else:
                taboo[i] = 1
                p = random.random()
                if p > 0.5 + 0.01 * i:
                    prev_solution = cand_solution
                    prev_time = cand_time
        print('bt: ', b_time)
        return best_solution
    def applyPopulation(self, population):

        futures = []
        for chrom in population:
            futures.append(self.pool.submit(self.applyChromTabu, chrom))

        wait(futures, return_when=ALL_COMPLETED)
        return [future.result() for future in futures]

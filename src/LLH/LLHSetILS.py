import time

from src.LLH.LLHUtils import timeTaken
from src.utils import encoding
from src.utils.parser import parse

# LLH方法类,维护局部搜索和扰动两类 LLH
# 维护一个全局最优解和当前邻域解
# 维护扰动LLH的评估数据


class LLHSetILS():
    def __init__(self):
        # 用于 VND 过程的 LLH
        self.vnd = []
        # 添加方法
        self.vnd.append(self.vnd1)

        # 用于扰动的 LLH
        self.shake = []
        #添加方法
        self.shake.append(self.shakeA)

        #根据 shake 长度初始化为全 0
        self.evaluation_recent_improve = [0 for i in range(len(self.shake))]
        self.evaluation_by_accept = [0 for i in range(len(self.shake))]
        self.evaluation_by_speed = []
        #

    #=======================工具方法===========================
    # 重设管理器状态
    def reset(self, problem_path):
        #算法开始运行时间
        self.time_start = time.time()
        #算法运行时间限制,单位为为毫秒
        self.time_limit = 1000 * 120
        # 重置问题参数
        self.parameters = parse(problem_path)
        # 初始化全局最优解
        self.best_solution = encoding.initializeResult(self.parameters)
        self.best_time = timeTaken(self.best_solution, self.parameters)
        #设置当前邻域解
        self.previous_solution = self.best_solution
        self.previous_time = self.best_time


    #已用时间,单位毫秒
    def elapsed_time(self):
        return (time.time() - self.time_start) * 1000

    # 检查超时
    def check_exceed_time_limit(self):
        return self.elapsed_time() > self.time_limit

    # 接受当前解
    def accept_proposal_solution(self, proposal_solution, proposal_time):
        self.previous_solution = proposal_solution
        self.previous_time = proposal_time
        #更新当前维护的全局最优解
        self.update_best_solution()

    # 更新当前邻域解为全局最优解
    def update_best_solution(self):
        if self.previous_time < self.best_time:
            self.best_solution = self.previous_solution
            self.best_time = self.previous_time

    def update_evaluation_score(self):
        pass





        # =====================VND局部搜索LLH============================
    # 2-opt
    def vnd1(self, current_solution):
        pass





    # ==========================扰动LLH==============================

    def shakeA(self, current_solution):
        (os, ms) = current_solution

        return (os, ms)
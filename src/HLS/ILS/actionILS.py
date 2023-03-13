# 选择-移动接受动作对
import math
import random
import time
from typing import Any, Tuple

import numpy as np

from src.LLH.LLHSetILS import LLHSetILS
from src.LLH.LLHUtils import timeTaken


class action:
    # 初始化
    def __init__(self):
        #移动接受方法组
        self.move_acceptors = []
        #添加所有函数到 move_acceptor
        self.move_acceptors.append(self.acceptanceOI)
        self.move_acceptors.append(self.acceptanceAM)
        self.move_acceptors.append(self.acceptanceNA)
        # self.move_acceptors.append(self.acceptanceSA)
        self.move_acceptors.append(self.acceptanceAPW)



        # LLH和解的管理器
        self.llh_manager = LLHSetILS()
        self.total_improvement = 0
        self.improvement_iter = 0
        # 选择-移动接受动作对, 一对索引的列表
        self.actions = [(i, j) for i in range(len(self.llh_manager.shake)) for j in range(len(self.move_acceptors))]

        # 算法运行时间限制,单位为为回合
        self.time_limit = 160
        # VND 运行时间参数
        self.NoE = 80
        self.VND_ITER = [10, 20, 40]



    # 执行动作
    def execute(self, action):
        # 选择的扰动LLH和移动接受方法
        perturbative_LLH = self.llh_manager.shake[self.actions[action][0]]
        Move_acceptance = self.move_acceptors[self.actions[action][1]]
        # VND local search运行计数
        counter = 0
        while counter < self.NoE:
            counter += 1
            # 扰动LLH应用到 llh_manager.previous_solution, 获得扰动后的解, 作为本次循环的起始
            current_solution = perturbative_LLH()
            # 解码
            current_time = timeTaken(current_solution, self.llh_manager.parameters)
            # 局部搜索
            proposal_solution, proposal_time = self.local_search(current_solution, current_time)
            # delta_f = proposal_time - self.llh_manager.previous_time
            # 移动接受
            Move_acceptance(proposal_solution, proposal_time)

        # print("local search run ", counter, " times")



    #局部搜索,遵循 VND 过程
    def local_search(self, current_solution, current_time):
        # 局部搜索算子重排序,长度为 llh_manager.vnd的长度
        ls_operators = [i for i in range(len(self.llh_manager.vnd))]
        random.shuffle(ls_operators)
        idx = 1
        #本循环内的解和时间
        proposal_solution = (current_solution[0].copy(), current_solution[1].copy())
        proposal_time = current_time
        while idx < len(ls_operators):
            # 选择算子
            operator = ls_operators[idx]
            # 生成新解,评估新解时间
            new_solution = self.llh_manager.vnd[operator](proposal_solution)
            new_time = timeTaken(new_solution, self.llh_manager.parameters)
            # 接受新解
            if proposal_time > new_time:
                self.total_improvement += (proposal_time - new_time)
                self.improvement_iter += 1
                proposal_solution = new_solution
                proposal_time = new_time
                idx = 1
            else:
                idx += 1

        # print('terminated! proposal: ', proposal_time)
        return proposal_solution, proposal_time



    # ==========================移动接受方法==========================
    # 仅改进
    def acceptanceOI(self, proposal_solution, proposal_time):
        if(self.llh_manager.previous_time > proposal_time):
            self.llh_manager.accept_proposal_solution(proposal_solution, proposal_time)
            return True
        else:
            return False

    # 随机游走,接受所有解
    def acceptanceAM(self, proposal_solution, proposal_time):
        self.llh_manager.accept_proposal_solution(proposal_solution, proposal_time)
        return True

    #接受改进解, 0.5 概率接受非改进解
    def acceptanceNA(self, proposal_solution, proposal_time):
        if (self.llh_manager.previous_time > proposal_time):
            self.llh_manager.accept_proposal_solution(proposal_solution, proposal_time)
            return True
        else:
            if random.random() < 0.5:
                self.llh_manager.accept_proposal_solution(proposal_solution, proposal_time)
                return True
            else:
                return False


    # 概率更差接受
    def acceptanceAPW(self, proposal_solution, proposal_time):
        T = 0.5
        delta_f = self.llh_manager.previous_time - proposal_time
        # print('total_improvement: ', self.total_improvement, ' improvement_iter: ', self.improvement_iter)
        if self.improvement_iter == 0:
            miu_impr = 0
        else:
            miu_impr = self.total_improvement / self.improvement_iter
        # 接收概率 p
        if miu_impr == 0:
            p = 1
        else:
            p = math.exp((delta_f / (T * miu_impr)))
        if random.random() < p:
            self.llh_manager.accept_proposal_solution(proposal_solution, proposal_time)
            return True
        else:
            return False

    # 模拟退火接受
    # def acceptanceSA(self, proposal_solution, proposal_time):
    #     T = 1
    #     t_max = self.iter_start + self.time_limit / self.NoE
    #     t_elapsed = self.elapsed_iter_time()
    #     delta_f = self.llh_manager.previous_time - proposal_time
    #     # print('total_improvement: ', self.total_improvement, ' improvement_iter: ', self.improvement_iter)
    #     if self.improvement_iter == 0:
    #         miu_impr = 0
    #     else:
    #         miu_impr = self.total_improvement / self.improvement_iter
    #     # 接收概率 p
    #     if miu_impr == 0:
    #         p = 1
    #     else:
    #         # 接受概率
    #         p = math.exp((delta_f / (T * miu_impr)) * (t_max / (t_max - t_elapsed)))
    #     if random.random() < p:
    #         self.llh_manager.accept_proposal_solution(proposal_solution, proposal_time)
    #         return True
    #     else:
    #         return False


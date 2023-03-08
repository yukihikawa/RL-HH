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
        self.move_acceptors.append(self.acceptanceNA)
        self.move_acceptors.append(self.acceptanceSA)

        # 选择-移动接受动作对, 一对索引的列表
        # LLH和解的管理器
        self.llh_manager = LLHSetILS()
        self.total_improvement = 0
        self.improvement_iter = 0
        #ILS 外循环
        self.out_iter = 0
        # 总循环
        self.total_iter = 0

    # 执行动作
    def execute(self, action):
        self.set_iter_start()
        # print(self.iter_start)
        t_expire = self.iter_start + self.time_limit / self.NoE # 本次迭代的时间限制
        # print('iter time: ', self.time_limit / self.NoE )
        # print(self.actions[action][0], ' and ', self.actions[action][1])
        # 选择的扰动LLH和移动接受方法
        Select_perturbative_LLH = self.selectors[self.actions[action][0]]
        # print('select: ', Select_perturbative_LLH)
        Move_acceptance = self.move_acceptors[self.actions[action][1]]
        # print('move acc: ', Move_acceptance)
        current = time.time()
        # 刷新上一次的解
        # self.llh_manager.refresh_previous_solution()
        # print('in execute')
        # print('global best: ', self.llh_manager.best_time,
        #       'previous: ', self.llh_manager.previous_time)
        while current < t_expire:
            # print('===========================================')
            # print('current time: ', current, 't_expire: ', t_expire)
            # print('prev_time:', self.llh_manager.previous_time)
            # 选择一个LLH应用到 llh_manager.previous_solution, 获得扰动后的解, 作为本次循环的起始
            current_solution, duration, shake_idx = Select_perturbative_LLH()
            # 解码
            current_time = timeTaken(current_solution, self.llh_manager.parameters)
            # print('shaked time: ', current_time)
            # 局部搜索
            # proposal_time: int | Any
            # proposal_solution: tuple[Any, Any] | None
            proposal_solution, proposal_time = self.local_search(current_solution, current_time)
            delta_f = proposal_time - self.llh_manager.previous_time
            # 移动接受
            accepted = Move_acceptance(proposal_solution, proposal_time)
            # print('accepted?', accepted)
            # print('new prev', self.llh_manager.previous_time)
            # 更新改进量和改进次数
            self.llh_manager.update_evaluation_score(shake_idx, duration, delta_f, accepted)
            current = time.time()

    #局部搜索,遵循 VND 过程
    def local_search_VND(self, current_solution, current_time):
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
                # print('improvement: ', proposal_time - new_time)
                # 更新改进量和改进次数
                self.total_improvement += (proposal_time - new_time)
                # print('total improvement: ', self.total_improvement)
                self.improvement_iter += 1
                # print('improvement iter: ', self.improvement_iter)
                proposal_solution = new_solution
                proposal_time = new_time

                # 重置算子顺序
                # random.shuffle(ls_operators)
                idx = 1
            else:
                # print("no improvement: ")
                idx += 1

        # print('terminated! proposal: ', proposal_time)
        return proposal_solution, proposal_time

    def local_search(self, current_solution, current_time, local_search_function, acceptance_function):

        (os, ms) = current_solution
        iter = 0
        iter_limit = len(os)
        # 搜索位置
        for i in range(len(os)):
            # 生成新解,评估新解时间
            new_solution = local_search_function(current_solution, i)
            new_time = timeTaken(new_solution, self.llh_manager.parameters)
            if acceptance_function(current_time - new_time):
                current_solution = new_solution
                current_time = new_time
        return current_solution, current_time





    # ==========================移动接受方法==========================
    # 仅改进,delta为旧时间- 新时间
    def acceptanceOI(self, delta, turn = 0):
        if delta > 0:
            return True
        else:
            return False

    #接受改进解, 0.5 概率接受非改进解
    def acceptanceNA(self, delta, turn = 0):
        if delta > 0:
            return True
        else:
            if random.random() < 0.5:
                return True
            else:
                return False

    # 模拟退火接受
    def acceptanceSA(self, delta, turn = 0):
        T = 1
        # print('total_improvement: ', self.total_improvement, ' improvement_iter: ', self.improvement_iter)
        if self.improvement_iter == 0:
            miu_impr = 0
        else:
            miu_impr = self.total_improvement / self.improvement_iter
        # 接收概率 p
        if miu_impr == 0:
            p = 1
        else:
            # 接受概率
            p = math.exp((delta / (T * miu_impr)) * (t_max / (t_max - t_elapsed)))
        if random.random() < p:
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


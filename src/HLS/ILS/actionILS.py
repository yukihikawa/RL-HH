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
        # 扰动 LLH 选择方法组
        self.selectors = []
        #添加所有函数到 selector
        self.selectors.append(self.selectorImprovement)
        self.selectors.append(self.selectorAccepted)
        self.selectors.append(self.selectorIOT)
        # self.selectors.append(self.selectorSpeed)
        # self.selectors.append(self.selectorSpeedAccepted)
        # self.selectors.append(self.selectorSpeedNew)

        #移动接受方法组
        self.move_acceptors = []
        #添加所有函数到 move_acceptor
        self.move_acceptors.append(self.acceptanceOI)
        self.move_acceptors.append(self.acceptanceAM)
        self.move_acceptors.append(self.acceptanceNA)
        # self.move_acceptors.append(self.acceptanceSA)
        self.move_acceptors.append(self.acceptanceAPW)

        # 选择-移动接受动作对, 一对索引的列表
        self.actions = [(i, j) for i in range(len(self.selectors)) for j in range(len(self.move_acceptors))]
        # LLH和解的管理器
        self.llh_manager = LLHSetILS()
        self.total_improvement = 0
        self.improvement_iter = 0

        # 算法开始运行时间
        self.time_start = time.time()
        # 算法运行时间限制,单位为为秒
        self.time_limit = 160
        self.iter_start = self.time_start

        self.NoE = 80

    # 设置单次迭代开始时间
    def set_iter_start(self):
        self.iter_start = time.time()

    # 本次迭代已用时间,单位毫秒
    def elapsed_iter_time(self):
        return time.time() - self.iter_start

    # 设置总体开始时间
    def set_time_start(self):
        self.time_start = time.time()
    # 总体已用时间,单位毫秒
    def elapsed_time(self):
        return time.time() - self.time_start

    # 检查总体超时
    def check_exceed_time_limit(self):
        return self.elapsed_time() > self.time_limit

    # 执行动作
    def execute(self, action):
        self.set_iter_start()
        # print(self.iter_start)
        # print(self.NoE)
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
        counter = 0
        while current < t_expire:
            counter += 1
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
        print("local search run ", counter, " times")



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


    # ========================扰动LLH选择方法===========================
    # 轮盘赌选择
    def roulette_selection(Self, data):
        # 计算概率分布
        total = sum(data)
        if total == 0:
            return random.randint(0, len(data) - 1)

        probabilities = [d / total for d in data]
        # 计算累计概率
        cum_probabilities = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]
        # print('cum_probabilities: ', cum_probabilities)
        selected_index = None
        rand = random.uniform(0, 1)
        for i, cum_prob in enumerate(cum_probabilities):
            if rand <= cum_prob:
                selected_index = i

                break
        return selected_index
    # 最近执行中的最大正向扰动
    def selectorImprovement(self):
        data = self.llh_manager.eval_recent_improve
        # 根据 data列表代表的概率分布,实现一个使用累计概率的轮盘赌选择
        idx = self.roulette_selection(data)
        t0 = time.time()
        # 选择一个LLH应用到 llh_manager.previous_solution, 获得扰动后的解
        current_solution = self.llh_manager.shake[idx]()
        t1 = time.time()
        return current_solution, t1 - t0, idx

    # 如果
    def selectorAccepted(self):
        data = self.llh_manager.eval_by_accept
        # 根据 data列表代表的概率分布,实现一个使用累计概率的轮盘赌选择
        idx = self.roulette_selection(data)
        t0 = time.time()
        # 选择一个LLH应用到 llh_manager.previous_solution, 获得扰动后的解
        current_solution = self.llh_manager.shake[idx]()
        t1 = time.time()
        return current_solution, t1 - t0, idx

    # 随时间改进 选择
    def selectorIOT(self):
        data = self.llh_manager.eval_improve_overtime
        # 根据 data列表代表的概率分布,实现一个使用累计概率的轮盘赌选择
        idx = self.roulette_selection(data)
        t0 = time.time()
        # 选择一个LLH应用到 llh_manager.previous_solution, 获得扰动后的解
        current_solution = self.llh_manager.shake[idx]()
        t1 = time.time()
        return current_solution, t1 - t0, idx

    def selectorSpeed(self):
        data = self.llh_manager.eval_by_speed
        # 根据 data列表代表的概率分布,实现一个使用累计概率的轮盘赌选择
        idx = self.roulette_selection(data)
        t0 = time.time()
        # 选择一个LLH应用到 llh_manager.previous_solution, 获得扰动后的解
        current_solution = self.llh_manager.shake[idx]()
        t1 = time.time()
        return current_solution, t1 - t0, idx

    def selectorSpeedAccepted(self):
        data = self.llh_manager.eval_by_speed_accepted
        # 根据 data列表代表的概率分布,实现一个使用累计概率的轮盘赌选择
        idx = self.roulette_selection(data)
        # print('idx: ', idx)
        t0 = time.time()
        # 选择一个LLH应用到 llh_manager.previous_solution, 获得扰动后的解
        current_solution = self.llh_manager.shake[idx]()
        t1 = time.time()
        return current_solution, t1 - t0, idx

    def selectorSpeedNew(self):
        data = self.llh_manager.eval_by_speed_new
        # 根据 data列表代表的概率分布,实现一个使用累计概率的轮盘赌选择
        idx = self.roulette_selection(data)
        t0 = time.time()
        # 选择一个LLH应用到 llh_manager.previous_solution, 获得扰动后的解

        current_solution = self.llh_manager.shake[idx]()
        t1 = time.time()
        return current_solution, t1 - t0, idx

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

    # 模拟退火接受
    def acceptanceSA(self, proposal_solution, proposal_time):
        T = 1
        t_max = self.iter_start + self.time_limit / self.NoE
        t_elapsed = self.elapsed_iter_time()
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
            # 接受概率
            p = math.exp((delta_f / (T * miu_impr)) * (t_max / (t_max - t_elapsed)))
        if random.random() < p:
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


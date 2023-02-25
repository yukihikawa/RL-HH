# 选择-移动接受动作对
import math
import random
import time

from src.LLH.LLHSetILS import LLHSetILS
from src.LLH.LLHUtils import timeTaken


class action:
    def __init__(self):
        # 扰动 LLH 选择方法组
        self.selector = []
        #添加所有函数到 selector
        self.selector.append(self.selectorImprovement)
        self.selector.append(self.selectorAccepted)
        self.selector.append(self.selectorIOT)
        self.selector.append(self.selectorSpeed)
        self.selector.append(self.selectorSpeedAccepted)
        self.selector.append(self.selectorSpeedNew)

        #移动接受方法组
        self.move_acceptor = []
        #添加所有函数到 move_acceptor
        self.move_acceptor.append(self.acceptanceOI)
        self.move_acceptor.append(self.acceptanceAM)
        self.move_acceptor.append(self.acceptanceNA)
        self.move_acceptor.append(self.acceptanceSA)
        self.move_acceptor.append(self.acceptanceAPW)

        # 选择-移动接受动作对, 一对索引的列表
        self.actions = [(i, j) for i in range(len(self.selector)) for j in range(len(self.move_acceptor))]
        # LLH和解的管理器
        self.llh_manager = LLHSetILS()
        self.total_improvement = 0
        self.improvement_iter = 0

    # 根据传入的 action选择一个动作对,
    def get_action(self, action):
        return self.actions[action][0], self.actions[action][1]

    # 包装方法,返回方法和运行时间
    def time_wrapper(self, func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            return result, end - start
        return wrapper


    #局部搜索,遵循 VND 过程
    def local_search(self, current_solution, current_time):
        # 局部搜索算子重排序,长度为 llh_manager.vnd的长度
        ls_operators = [i for i in range(len(self.llh_manager.vnd))]
        random.shuffle(ls_operators)
        #
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
                proposal_solution = new_solution
                proposal_time = new_time
                #更新改进量和改进次数
                self.total_improvement += (proposal_time - new_time)
                self.improvement_iter += 1
                # 重置算子顺序
                random.shuffle(ls_operators)
                idx = 1
            else:
                idx += 1
        return proposal_solution, proposal_time

    # ========================扰动LLH选择方法===========================
    # 最近执行中的最大正向扰动
    def selectorImprovement(self):
        # 选择一个LLH
        pass

    # 如果
    def selectorAccepted(self):
        # 选择一个LLH
        pass

    # 随时间改进 选择
    def selectorIOT(self):
        # 选择一个LLH
        pass

    def selectorSpeed(self):
        # 选择一个LLH
        pass

    def selectorSpeedAccepted(self):
        # 选择一个LLH
        pass

    def selectorSpeedNew(self):
        # 选择一个LLH
        pass

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
        t_max =self.llh_manager.time_limit
        t_elapsed = self.llh_manager.elapsed_time()
        delta_f = self.llh_manager.previous_time - proposal_time
        miu_impr = self.improvement_iter / self.total_improvement
        # 接受概率
        p = math.exp((delta_f / (T * miu_impr)) * (t_max / (t_max - t_elapsed)))
        if random.random() < p:
            self.llh_manager.accept_proposal_solution(proposal_solution, proposal_time)
            return True
        else:
            return False

    # 概率更差接受
    def acceptanceAPW(self, proposal_solution, proposal_time):
        T = 1
        delta_f = self.llh_manager.previous_time - proposal_time
        miu_impr = self.total_improvement / self.improvement_iter
        # 接收概率 p
        p = math.exp((delta_f / (T * miu_impr)))
        if random.random() < p:
            self.llh_manager.accept_proposal_solution(proposal_solution, proposal_time)
            return True
        else:
            return False


# 选择-移动接受动作对
import math
import random
import time
from typing import Any, Tuple

import numpy as np

from src.LLH.LLHSetILS import LLHSetILS
from src.LLH.LLHUtils import timeTaken, get_machine_workload, getMachineIdx
from src.utils.decoding import decode, split_ms
from src.utils.encoding import initializeResult
from src.utils.parser import parse


class ILSSet:
    # 初始化
    def __init__(self):

        # 局部搜索方法组
        self.local_search = []
        # 添加所有函数到 local_search
        self.local_search.append(self.local_search1)
        self.local_search.append(self.local_search2)
        self.local_search.append(self.local_search3)

        # 扰动方法组
        self.perturbation = []
        # 添加所有函数到 perturbation
        self.perturbation.append(self.perturbation1)
        # self.perturbation.append(self.perturbation2)
        # self.perturbation.append(self.vnd13)
        self.perturbation.append(self.vnd14)
        self.perturbation.append(self.vnd14_1)
        self.perturbation.append(self.shakeF)


        # 选择-移动接受动作对, 一对索引的列表,
        # 将local_search和perturbation的索引组合成一对元组，加入到actions中
        self.actions = []
        for i in range(len(self.local_search)):
            for j in range(len(self.perturbation)):
                self.actions.append((i, j))


    # 重设管理器状态
    def reset(self, problem_path, max_iter = 10000):
        self.max_iter = max_iter
        self.prev_iter = 0
        self.total_fitness = 0
        # 重置问题参数
        self.parameters = parse(problem_path)
        self.best_solution = self.previous_solution = initializeResult(self.parameters)
        self.best_time = self.previous_time = timeTaken(self.previous_solution, self.parameters)

    # 模拟退火接受策略
    def acceptanceSA(self, delta, temperature = 1000, scale = 0.1, alpha = 0.9954):
        # delta 为新时间 - 旧时间
        if delta < 0:
            return True
        else:
            p = math.exp((-1) / (temperature * (alpha ** self.prev_iter))) * scale
            if random.random() < p:
                return True
            else:
                return False


    def acceptanceSA_inner(self, delta, pi, temperature = 1000, scale = 0.1, alpha = 0.85):
        # delta 为新时间 - 旧时间
        if delta < 0:
            return True
        else:
            p = math.exp((-1) / (temperature * (alpha ** pi))) * scale
            if random.random() < p:
                return True
            else:
                return False
    # 执行动作
    def ILS(self, action):
        self.prev_iter += 1
        # 当前解的时间
        o_time = self.previous_time
        #对self.previous_solution进行扰动

        p_solution = self.perturbation[self.actions[action][1]]()

        #对p_solution进行局部搜索
        l_solution = self.local_search[self.actions[action][0]](p_solution)
        l_time = timeTaken(l_solution, self.parameters)
        # print("ori time: ", o_time, " perturb time: ", l_time)
        # 更新最优解
        if self.best_time > l_time:
            self.best_solution = l_solution
            self.best_time = l_time
        #接受新解
        if self.acceptanceSA(l_time - o_time):
            self.previous_solution = l_solution
            self.previous_time = l_time
        # 保存历史适应度
        self.total_fitness += self.previous_time
        # 返回是否改进了当前解
        return l_time - o_time

    pass
    #多重ILS过程
    def multi_ILS(self, action, G_MAX = 50):
        self.prev_iter += 1
        o_time = self.previous_time
        for i in range(G_MAX):
            # 当前解的时间
            # o_time = self.previous_time
            # 对self.previous_solution进行扰动

            p_solution = self.perturbation[self.actions[action][1]]()
            # 对p_solution进行局部搜索
            l_solution = self.local_search[self.actions[action][0]](p_solution)
            l_time = timeTaken(l_solution, self.parameters)
            print("ori time: ", self.previous_time, " perturb time: ", l_time)
            # 更新最优解
            if self.best_time > l_time:
                self.best_solution = l_solution
                self.best_time = l_time
            # 接受新解
            if self.acceptanceSA_inner(l_time - self.previous_time, i):
                self.previous_solution = l_solution
                self.previous_time = l_time
        # 保存历史适应度
        self.total_fitness += self.previous_time
        # 返回是否改进了当前解
        return self.previous_time - o_time


    # 局部搜索LLH========================================================
    #  1 随机工序码插入局部搜索
    def local_search1(self, current_solution):
        (os, ms) = current_solution
        current_time = timeTaken(current_solution, self.parameters)
        # 选择一位工序码
        idx = random.randint(0, len(os) - 1)
        for i in range(len(os)):
            newOs = os.copy()
            k = newOs[idx]
            newOs = newOs[0:idx] + newOs[idx + 1: len(newOs)]
            newOs = newOs[0: i] + [k] + newOs[i: len(newOs)]
            if current_time > timeTaken((newOs, ms), self.parameters):
                return (newOs, ms)
        return (os, ms)

    # 2 机器码局部搜索
    def local_search2(self, current_solution):
        (os, ms) = current_solution
        current_time = timeTaken(current_solution, self.parameters)
        # 获取作业集合
        jobs = self.parameters['jobs']
        # 搜索整个机器码序列
        for idx in range(0, len(ms)):
            mcLength = 0  # 工具人
            jobIdx = -1  # 所属工作号
            for job in jobs:
                jobIdx += 1
                if mcLength + len(job) >= idx + 1:
                    break
                else:
                    mcLength += len(job)
            opIdx = idx - mcLength  # 指定位置对应的 在工件中的工序号
            # 开始搜索机器集合
            for i in range(0, len(jobs[jobIdx][opIdx])):
                newMs = ms.copy()
                newMs[idx] = i
                new_time = timeTaken((os, newMs), self.parameters)
                if current_time > new_time:
                    return (os, newMs)
        return (os, ms)

    # 3 并行局部搜索
    def local_search3(self, current_solution):
        (os, ms) = current_solution
        current_time = timeTaken(current_solution, self.parameters)
        # 获取作业集合
        jobs = self.parameters['jobs']
        # 选择一位工序码
        idx = random.randint(0, len(os) - 1)

        for i in range(0, len(os)):
            newOs = os.copy()
            k = newOs[idx]
            newOs = newOs[0:idx] + newOs[idx + 1: len(newOs)]
            newOs = newOs[0: i] + [k] + newOs[i: len(newOs)]
            # 工序新位置到位
            # 开始机器码搜索
            machineIdx = getMachineIdx(i, newOs, self.parameters)
            mcLength = 0  # 工具人
            jobIdx = -1  # 所属工作号
            # 获取作业编号
            for job in jobs:
                jobIdx += 1
                if mcLength + len(job) >= machineIdx + 1:
                    break
                else:
                    mcLength += len(job)
            opIdx = machineIdx - mcLength
            for j in range(0, len(jobs[jobIdx][opIdx])):
                newMs = ms.copy()
                newMs[machineIdx] = j
                new_time = timeTaken((newOs, newMs), self.parameters)
                if current_time > new_time:
                    return (newOs, newMs)
        return (os, ms)



    # 13 工作负载邻域
    def vnd13(self, current_solution):
        # 对当前解进行解码
        machine_operation = decode(self.parameters, current_solution[0], current_solution[1])
        # 获取工作负载
        workload = get_machine_workload(self.parameters, machine_operation)
        # 取得最大负载机器,workload中最大值的索引, 从 0 开始的
        max_workload_machine = workload.index(max(workload))
        # 从具有最大负载的机器中随机选择一个工序
        selected_op = random.choice(machine_operation[max_workload_machine])
        # 获取工序信息
        job_idx, op_idx = map(int, selected_op[0].split('-'))
        op_idx -= 1
        # 获取工序的机器集合
        machine_set = self.parameters['jobs'][job_idx][op_idx]
        # 当前工序所在机器负载
        prev_load = max(workload)
        # 从机器集合中选择负载最小的机器
        selected_new_machine = 0
        for i in range(len(machine_set)):  # 遍历机器合集
            machine_idx = machine_set[i]['machine']
            new_load = workload[machine_idx - 1]
            if new_load < prev_load:
                prev_load = new_load
                selected_new_machine = i  # 新的ms编码
        #         print("sdfgsd:", selected_new_machine)
        # print('selected_new_machine ggggg: ', selected_new_machine)
        # 生成新的ms编码
        ms_s = split_ms(self.parameters, current_solution[1])  # 分离的ms编码
        # 在 ms 中的位置
        ms_idx = 0
        for i in range(job_idx):
            ms_idx += len(ms_s[i])
        ms_idx += op_idx
        new_ms = current_solution[1].copy()
        # print('old ms: ', new_ms[ms_idx])
        new_ms[ms_idx] = selected_new_machine
        return (current_solution[0], new_ms)


    # 扰动LLH========================================================
    pass
    # 15 作业交换移动
    def perturbation1(self):
        (oos, oms) = self.previous_solution
        os = oos.copy()
        ms = oms.copy()
        jobs = self.parameters['jobs']
        # 选择两个不同的作业
        job1 = job2 =  random.randint(0, len(jobs) - 1)
        while job1 == job2:
            job2 = random.randint(0, len(jobs) - 1)
        idx1 = idx2 = 0
        while idx1 < len(os) and idx2 < len(os):
            while idx1 < len(os) and os[idx1] != job1:
                idx1 += 1
            while idx2 < len(os) and os[idx2] != job2:
                idx2 += 1
            if idx1 < len(os) and idx2 < len(os) and os[idx1] == job1 and os[idx2] == job2:
                os[idx1], os[idx2] = os[idx2], os[idx1]
            idx1 += 1
            idx2 += 1
        return (os, ms)

    # 2 区间内工序码打乱
    def perturbation2(self):
        (os, ms) = self.previous_solution
        ida = idb = random.randint(0, len(os) - 2)
        while ida == idb:
            idb = random.randint(0, len(os) - 1)
        if ida > idb:
            ida, idb = idb, ida
        # 打乱区间的工序码顺序
        newOs = os.copy()
        mid = newOs[ida:idb + 1]
        random.shuffle(mid)
        newOs = os[:ida] + mid + os[idb + 1:]
        # 替换当前解
        return (newOs, ms)



    # 14 swap 移动邻域，前后交换
    def vnd14(self):
        G_max = 3
        (os, ms) = self.previous_solution
        new_os = os.copy()
        for i in range(G_max):
            idx = random.randint(1, len(os) - 2)
            if new_os[idx] != new_os[idx - 1]:
                new_os[idx - 1], new_os[idx] = new_os[idx], new_os[idx - 1]
            else:
                new_os[idx], new_os[idx + 1] = new_os[idx + 1], new_os[idx]
        return (new_os, ms)

    def vnd14_1(self):
        G_max = 5
        (os, ms) = self.previous_solution
        new_os = os.copy()
        for i in range(G_max):
            idx = random.randint(1, len(os) - 2)
            if new_os[idx] != new_os[idx - 1]:
                new_os[idx - 1], new_os[idx] = new_os[idx], new_os[idx - 1]
            else:
                new_os[idx], new_os[idx + 1] = new_os[idx + 1], new_os[idx]
        return (new_os, ms)

    # 工序码子序列逆序
    def shakeF(self):
        (os, ms) = self.previous_solution
        ida = idb = random.randint(0, len(os) - 1)
        while ida == idb:
            idb = random.randint(0, len(os) - 1)
        if ida > idb:
            ida, idb = idb, ida
        newOs = os.copy()
        newOs[ida:idb + 1] = newOs[ida:idb + 1][::-1]
        return (newOs, ms)
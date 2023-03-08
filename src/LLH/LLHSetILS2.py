import random
import sys
import time

from src.LLH.LLHUtils import timeTaken, changeMsRandom, getMachineIdx, get_machine_workload
from src.utils import encoding
from src.utils.decoding import decode, split_ms
from src.utils.parser import parse

# LLH方法类,维护局部搜索和扰动两类 LLH
# 维护一个全局最优解和当前邻域解
# 维护扰动LLH的评估数据



class LLHSetILS():
    def __init__(self):
        # 用于 VND 过程的 LLH
        self.vnd = []


        # 用于扰动的 LLH
        self.shake = []
        #添加方法

    #=======================工具方法===========================


    # =====================局部搜索LLH============================
    # 1 工序码插入，单次动作
    def ls_piece1(self, start_position, to_position, current_solution):
        (os, ms) = current_solution
        current_time = timeTaken(current_solution, self.parameters)
        # 选择一位工序码
        idx = start_position
        newOs = os.copy()
        k = newOs[idx]
        newOs = newOs[0:idx] + newOs[idx + 1: len(newOs)]
        newOs = newOs[0: to_position] + [k] + newOs[to_position: len(newOs)]
        if current_time > timeTaken((newOs, ms), self.parameters):
            return (newOs, ms)
        return (os, ms)

    # 10 工序码局部搜索
    def vnd10(self, current_solution):
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

    # 11 机器码局部搜索
    def vnd11(self, current_solution):
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

    # 12 并行局部搜索
    def vnd12(self, current_solution):
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


    # 14 swap 移动邻域，前后交换
    def vnd14(self, current_solution):
        G_max = 3
        (os, ms) = current_solution
        new_os = os.copy()
        for i in range(G_max):
            idx = random.randint(1, len(os) - 2)
            if new_os[idx] != new_os[idx - 1]:
                new_os[idx - 1], new_os[idx] = new_os[idx], new_os[idx - 1]
            else:
                new_os[idx], new_os[idx + 1] = new_os[idx + 1], new_os[idx]
        return (new_os, ms)
    def vnd14_1(self, current_solution):
        G_max = 5
        (os, ms) = current_solution
        new_os = os.copy()
        for i in range(G_max):
            idx = random.randint(1, len(os) - 2)
            if new_os[idx] != new_os[idx - 1]:
                new_os[idx - 1], new_os[idx] = new_os[idx], new_os[idx - 1]
            else:
                new_os[idx], new_os[idx + 1] = new_os[idx + 1], new_os[idx]
        return (new_os, ms)

    # 15 作业交换移动
    def vnd15(self, current_solution):
        (os, ms) = current_solution
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


    def vnd18(self, current_solution):
        (os, ms) = current_solution
        jobs = self.parameters['jobs']
        new_os = os.copy()

        # 选择两个不同的作业
        job1 = job2 = random.randint(0, len(jobs) - 1)
        while job1 == job2:
            job2 = random.randint(0, len(jobs) - 1)
        idx1 = idx2 = 0
        while idx1 < len(new_os) and idx2 < len(new_os):
            while idx1 < len(new_os) and new_os[idx1] != job1:
                idx1 += 1
            while idx2 < len(new_os) and new_os[idx2] != job2:
                idx2 += 1
            if idx1 < len(new_os) and idx2 < len(new_os) and new_os[idx1] == job1 and new_os[idx2] == job2:
                new_os[idx1], new_os[idx2] = new_os[idx2], new_os[idx1]
            idx1 += 1
            idx2 += 1

        return (new_os, ms)





    # ==========================扰动LLH==============================

    # A 随机工序码区间破坏
    def shakeA(self, current_solution):
        (os, ms) = current_solution
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

    # B 工序码随机交换同时随机改变对应位置机器码
    def shakeB(self):
        (os, ms) = self.previous_solution
        newOs = os.copy()
        newMs = ms.copy()
        # 随机选择两个工序码
        ida = idb = random.randint(0, len(os) - 1)
        while ida == idb:
            idb = random.randint(0, len(os) - 1)
        # 交换工序码
        newOs[ida], newOs[idb] = newOs[idb], newOs[ida]
        # 定位机器码
        machineIda = getMachineIdx(ida, os, self.parameters)
        machineIdb = getMachineIdx(idb, os, self.parameters)
        # 随机改变机器码
        newMs = changeMsRandom(machineIda, newMs, self.parameters)
        newMs = changeMsRandom(machineIdb, newMs, self.parameters)
        return (newOs, newMs)

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

    # 14 多重 swap 移动
    def shakeG(self):
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

    # 作业交换
    def shakeH(self):
        (os, ms) = self.previous_solution
        jobs = self.parameters['jobs']
        new_os = os.copy()

        # 选择两个不同的作业
        job1 = job2 = random.randint(0, len(jobs) - 1)
        while job1 == job2:
            job2 = random.randint(0, len(jobs) - 1)
        idx1 = idx2 = 0
        while idx1 < len(new_os) and idx2 < len(new_os):
            while idx1 < len(new_os) and new_os[idx1] != job1:
                idx1 += 1
            while idx2 < len(new_os) and new_os[idx2] != job2:
                idx2 += 1
            if idx1 < len(new_os) and idx2 < len(new_os) and new_os[idx1] == job1 and new_os[idx2] == job2:
                new_os[idx1], new_os[idx2] = new_os[idx2], new_os[idx1]
            idx1 += 1
            idx2 += 1

        return (new_os, ms)
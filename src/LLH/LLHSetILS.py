import random
import sys
import time

from src.LLH.LLHUtils import timeTaken, changeMsRandom, getMachineIdx
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
        self.vnd.append(self.vnd2)
        self.vnd.append(self.vnd3)
        self.vnd.append(self.vnd4)
        self.vnd.append(self.vnd5)
        self.vnd.append(self.vnd6)
        self.vnd.append(self.vnd7)
        self.vnd.append(self.vnd8)
        self.vnd.append(self.vnd9)
        self.vnd.append(self.vnd10)
        self.vnd.append(self.vnd11)
        self.vnd.append(self.vnd12)


        # 用于扰动的 LLH
        self.shake = []
        #添加方法
        self.shake.append(self.shakeA)
        self.shake.append(self.shakeB)
        self.shake.append(self.shakeC)
        self.shake.append(self.shakeD)
        self.shake.append(self.shakeE)
        self.shake.append(self.shakeF)


        #评估数据
        self.total_duration = [0] * len(self.shake)#运行总时间
        self.total_improvement = [0] * len(self.shake) #获得改进的量
        self.total_Noop = [0] * len(self.shake)  #无变化次数
        self.total_selected = [0] * len(self.shake) #被选择次数
        self.total_accepted = [0] * len(self.shake) #被接受次数
        self.C = 100000
        #根据 shake 长度初始化为全 0
        self.eval_recent_improve = [1000] * len(self.shake) #最近的改进
        self.eval_by_accept = [1000] * len(self.shake) #最近被接受
        self.eval_improve_overtime = [sys.float_info.max] * len(self.shake) #平均时间单位的改进
        self.eval_by_speed = [sys.float_info.max] * len(self.shake) #运行速度->平均运行时间的倒数
        self.eval_by_speed_accepted = [sys.float_info.max] * len(self.shake) #获得一个被接受解所需的平均运行时间的倒数
        self.eval_by_speed_new = [sys.float_info.max] * len(self.shake) #获得一个变动解所需的平均运行时间的倒数
        #

    #=======================工具方法===========================
    # 重设管理器状态
    def reset(self, problem_path):

        # 重置问题参数
        self.parameters = parse(problem_path)
        # 初始化全局最优解
        self.best_solution = encoding.initializeResult(self.parameters)
        self.best_time = timeTaken(self.best_solution, self.parameters)
        #设置当前邻域解
        self.previous_solution = self.best_solution
        self.previous_time = self.best_time

        # 重置评估数据
        self.total_duration = [0] * len(self.shake)  # 运行总时间
        self.total_improvement = [0] * len(self.shake)  # 获得改进的量
        self.total_Noop = [0] * len(self.shake)  # 无变化次数
        self.total_selected = [0] * len(self.shake)  # 被选择次数
        self.total_accepted = [0] * len(self.shake)  # 被接受次数
        self.C = 100000
        # 根据 shake 长度初始化
        self.eval_recent_improve = [1000] * len(self.shake)  # 最近的改进
        self.eval_by_accept = [1000] * len(self.shake)  # 最近被接受
        self.eval_improve_overtime = [100000000000.0] * len(self.shake)  # 平均时间单位的改进
        self.eval_by_speed = [10000000.0] * len(self.shake)  # 运行速度->平均运行时间的倒数
        self.eval_by_speed_accepted = [10000000.0] * len(self.shake)  # 获得一个被接受解所需的平均运行时间的倒数
        self.eval_by_speed_new = [10000000.0] * len(self.shake)  # 获得一个变动解所需的平均运行时间的倒数

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

    def refresh_previous_solution(self):
        self.previous_solution = self.best_solution
        self.previous_time = self.best_time

    # 根据被选择的llh序号\运行持续时间\改进量\是否被接受,更新评估数据sja
    def update_evaluation_score(self, shake_idx: int, duration: float, delta_f: int, accepted: bool):
        #更新扰动评估基础数据
        self.total_duration[shake_idx] += duration
        self.total_improvement[shake_idx] += max(0, delta_f)
        if delta_f == 0:
            self.total_Noop[shake_idx] += 1
        self.total_selected[shake_idx] += 1
        if accepted:
            self.total_accepted[shake_idx] += 1
        #更新选择依赖数据
        self.eval_recent_improve[shake_idx] = max(0, delta_f)
        self.eval_by_accept[shake_idx] = 1 if accepted else 0
        self.eval_improve_overtime[shake_idx] = self.C * self.total_improvement[shake_idx]  / self.total_duration[shake_idx]
        self.eval_by_speed[shake_idx] = (self.total_selected[shake_idx] + 1) / self.total_duration[shake_idx]
        self.eval_by_speed_accepted[shake_idx] = (self.total_accepted[shake_idx] + 1) / self.total_duration[shake_idx]
        self.eval_by_speed_new[shake_idx] = (self.total_accepted[shake_idx] - self.total_Noop[shake_idx] + 1) / self.total_duration[shake_idx]


    # =====================VND局部搜索LLH============================
    # 1-opt
    # 1 选中一位工序码,随机插入工序码其他位置
    def vnd1(self, current_solution):
        (os, ms) = current_solution
        idx = random.randint(0, len(os) - 1)
        new_idx = idx
        while new_idx == idx:
            new_idx = random.randint(0, len(os) - 2)
        # 插入新位置
        new_os = os.copy()
        k = os[idx]
        del new_os[idx]
        new_os.insert(new_idx, k)
        return (new_os, ms)

    # 2 选中一位机器码,随机改为可用机器集中的其他机器码
    def vnd2(self, current_solution):
        (os, ms) = current_solution
        machineIdx = random.randint(0, len(ms) - 1)
        return (os, changeMsRandom(machineIdx, ms, self.parameters))

    # 3 移动单个工序码,并改变其机器码
    def vnd3(self, current_solution):
        (os, ms) = current_solution
        idx = random.randint(0, len(os) - 1)
        new_idx = idx
        while new_idx == idx:
            new_idx = random.randint(0, len(os) - 2)
        # 插入新位置
        new_os = os.copy()
        k = os[idx]
        del new_os[idx]
        new_os.insert(new_idx, k)
        # 改变机器码
        machineIdx = getMachineIdx(new_idx, os, self.parameters)
        return (new_os, changeMsRandom(machineIdx, ms, self.parameters))

    # 4 随机交换两个工序码
    def vnd4(self, current_solution):
        (os, ms) = current_solution
        idx2 = idx1 = random.randint(0, len(os) - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, len(os) - 1)
        new_os = os.copy()
        new_os[idx1], new_os[idx2] = new_os[idx2], new_os[idx1]
        return (new_os, ms)

    # 5 随机反转工序码子序列
    def vnd5(self, current_solution):
        (os, ms) = current_solution
        idx2 = idx1 = random.randint(0, len(os) - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, len(os) - 1)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        new_os = os.copy()
        new_os[idx1:idx2] = new_os[idx1:idx2][::-1]
        return (new_os, ms)

    # 6 随机前移工序码子序列
    def vnd6(self, current_solution):
        (os, ms) = current_solution
        ida = idb = random.randint(0, len(os) - 1)
        while ida == idb:
            idb = random.randint(0, len(os) - 1)
        if ida > idb:
            ida, idb = idb, ida
        newOs = os[ida:idb + 1] + os[:ida] + os[idb + 1:]

        return (newOs, ms)

    # 7 工序码随机交换同时随机改变对应位置机器码
    def vnd7(self, current_solution):
        (os, ms) = current_solution
        new_os = os.copy()
        new_ms = ms.copy()
        # 交换工序码
        idx2 = idx1 = random.randint(0, len(os) - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, len(os) - 1)
        new_os[idx1], new_os[idx2] = new_os[idx2], new_os[idx1]
        # 改变机器码
        machineIdx1 = getMachineIdx(idx1, os, self.parameters)
        machineIdx2 = getMachineIdx(idx2, os, self.parameters)
        new_ms = changeMsRandom(machineIdx1, new_ms, self.parameters)
        new_ms = changeMsRandom(machineIdx2, new_ms, self.parameters)
        return (new_os, new_ms)

    # 8 工序码随机反转子序列并同时随机改变对应位置机器码
    def vnd8(self, current_solution):
        (os, ms) = current_solution
        idx2 = idx1 = random.randint(0, len(os) - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, len(os) - 1)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        new_os = os.copy()
        new_os[idx1:idx2 + 1] = new_os[idx1:idx2 + 1][::-1]
        new_ms = ms.copy()
        # 改变机器码
        for idx in range(idx1, idx2 + 1):
            machineIdx = getMachineIdx(idx, new_os, self.parameters)
            new_ms = changeMsRandom(machineIdx, new_ms, self.parameters)
        return (new_os, new_ms)

    # 9 工序码随机前移子序列并同时随机改变对应位置机器码
    def vnd9(self, current_solution):
        (os, ms) = current_solution
        ida = idb = random.randint(0, len(os) - 1)
        while ida == idb:
            idb = random.randint(0, len(os) - 1)
        if ida > idb:
            ida, idb = idb, ida
        new_os = os[ida:idb + 1] + os[:ida] + os[idb + 1:]
        new_ms = ms.copy()
        # 改变机器码
        for idx in range(ida, idb + 1):
            machineIdx = getMachineIdx(idx, new_os, self.parameters)
            new_ms = changeMsRandom(machineIdx, new_ms, self.parameters)
        return (new_os, new_ms)

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

    # ==========================扰动LLH==============================

    # A 随机工序码区间破坏
    def shakeA(self):
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

    # C 随机改变单个机器码
    def shakeC(self):
        (os, ms) = self.previous_solution
        newMs = ms.copy()
        # 随机选择一个机器码
        machineIdx = random.randint(0, len(ms) - 1)
        # 随机改变机器码
        newMs = changeMsRandom(machineIdx, newMs, self.parameters)
        return (os, newMs)

    # 随机交换两个工序码
    def shakeD(self):
        (os, ms) = self.previous_solution
        newOs = os.copy()
        # 随机选择两个工序码
        ida = idb = random.randint(0, len(os) - 1)
        while ida == idb:
            idb = random.randint(0, len(os) - 1)
        # 交换工序码
        newOs[ida], newOs[idb] = newOs[idb], newOs[ida]
        return (newOs, ms)

    # 随机工序码区间破坏
    def shakeE(self):
        (os, ms) = self.previous_solution
        ida = idb = random.randint(0, len(os) - 1)
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
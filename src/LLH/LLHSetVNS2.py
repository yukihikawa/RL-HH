import random

from src.LLH.LLHUtils import timeTaken, getMachineIdx, changeMsRandom
from src.utils import encoding
from src.utils.parser import parse


class LLHSetVNS():
    # 初始化
    def __init__(self, train = False):
        # 是否为训练环境
        self.train = train

        #llh
        self.llh = []
        # 添加所有函数到 llh
        self.llh.append(self.heuristic1)
        self.llh.append(self.heuristic2A)
        self.llh.append(self.heuristic3A)
        self.llh.append(self.heuristic4)
        self.llh.append(self.heuristic5)
        self.llh.append(self.heuristic6)
        # self.llh.append(self.heuristicA)
        self.llh.append(self.heuristicB)
        # self.llh.append(self.heuristicC)
        self.llh.append(self.heuristicD)
        self.llh.append(self.heuristicE)
        self.llh.append(self.heuristicF)



    # 工具方法=======================================
    # 更新最优解
    def update_best_solution(self):
        if self.best_time > self.previous_time:
            self.best_solution = self.previous_solution
            self.best_time = self.previous_time

    # 环境重设
    def reset(self, problem_path):
        self.parameters = parse(problem_path)

        if self.train:
            self.best_solution = self.previous_solution = encoding.initializeFixedResult(self.parameters)
        else:
            self.best_solution = self.previous_solution = encoding.initializeResult(self.parameters)
        self.previous_time = self.best_time = timeTaken(self.previous_solution, self.parameters)

    # 区别接受包装器
    def accept_wrapper(self, llh_call):
        new_solution, new_time = self.llh[llh_call]()
        # 当前解更新策略
        # VND LLH: 接受改进
        if llh_call in [0, 1, 2, 3, 4]:
            if new_time < self.previous_time:
                self.previous_solution = new_solution
                self.previous_time = new_time
        # SHAKING LLH: 直接接受
        else:
            self.previous_solution = new_solution
            self.previous_time = new_time
        # 最优解更新策略
        self.update_best_solution()




    # VND==============================================
    # 工序码局部搜索, 改写完成
    def heuristic1(self):
        (os, ms) = self.previous_solution
        idx = random.randint(0, len(os) - 1)
        for i in range(0, len(os)):
            newOs = os.copy()
            k = newOs[idx]
            newOs = newOs[0:idx] + newOs[idx + 1: len(newOs)]
            newOs = newOs[0: i] + [k] + newOs[i: len(newOs)]
            new_time = timeTaken((newOs, ms), self.parameters)
            if self.previous_time > new_time:
                return (newOs, ms), new_time
        return (os, ms), self.previous_time

    # 机器码局部搜索，全搜一遍,返回新解与时间
    def heuristic2(self):
        (os, ms) = self.previous_solution
        jobs = self.parameters['jobs']
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
                if self.previous_time > new_time:
                    return (os, newMs), new_time
        return (os, ms), self.previous_time

    # 机器码局部搜索,随机替换
    def heuristic2A(self):
        (os, ms) = self.previous_solution
        tms = ms.copy()
        t_time = self.previous_time
        for i in range(0, len(tms)):
            newMs = changeMsRandom(i, tms, self.parameters)
            new_time = timeTaken((os, newMs), self.parameters)
            if self.previous_time > new_time:
                tms, t_time = newMs, new_time
        return (os, tms), t_time

    # 并行局部搜索,搜索 OS, 搜索 MS 改进在此
    def heuristic3(self):
        (os, ms) = self.previous_solution
        # 获取作业集合
        jobs = self.parameters['jobs']
        idx = random.randint(0, len(os) - 1)
        # 获取作业编号
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
            for job in jobs:
                jobIdx += 1
                if mcLength + len(job) >= machineIdx + 1:
                    break
                else:
                    mcLength += len(job)
            # 作业内部的工序号
            opIdx = machineIdx - mcLength
            for j in range(0, len(jobs[jobIdx][opIdx])):
                newMs = self.previous_solution[1].copy()
                newMs[machineIdx] = j
                new_time = timeTaken((newOs, newMs), self.parameters)
                if self.previous_time > new_time:
                    return (newOs, newMs), new_time
        return (os, ms), self.previous_time

    # 搜索 OS,随机改变对应位置机器码
    def heuristic3A(self):
        (os, ms) = self.previous_solution
        tos = os.copy()
        tms = ms.copy()
        t_time = self.previous_time
        # print(tos)
        idx = random.randint(0, len(tos) - 1)

        # print('selected position: ', idx)
        for i in range(0, len(tos)):
            newOs = tos.copy()
            k = newOs[idx]
            newOs = newOs[0:idx] + newOs[idx + 1: len(newOs)]
            newOs = newOs[0: i] + [k] + newOs[i: len(newOs)]
            machineIdx = getMachineIdx(i, os, self.parameters)
            newMs = changeMsRandom(machineIdx, ms, self.parameters)
            # print(newOs)
            new_time = timeTaken((newOs, newMs), self.parameters)
            if self.previous_time > new_time:
                tos = newOs
                tms = newMs
                t_time = new_time
        return (tos, tms), t_time

    # 10. 随机前移工序码子序列, 并改变对应位置的机器码
    def heuristic4(self):
        (os, ms) = self.previous_solution
        ida = idb = random.randint(0, len(os) - 2)
        while ida == idb:
            idb = random.randint(0, len(os) - 1)
        if ida > idb:
            ida, idb = idb, ida
        newOs = os.copy()
        newOs = newOs[ida:idb + 1] + newOs[:ida] + newOs[idb + 1:]
        newMs = ms.copy()
        for i in range(0, idb - ida + 1):
            newMs = changeMsRandom(i, newMs, self.parameters)
        new_time = timeTaken((newOs, newMs), self.parameters)
        if self.previous_time > new_time:
            return (newOs, newMs), new_time
        return (os, ms), self.previous_time

    # 5. 随机改变单个机器码
    def heuristic5(self):
        (os, ms) = self.previous_solution
        machineIdx = random.randint(0, len(ms) - 1)
        # ('selected idx : ', machineIdx)
        newMs = changeMsRandom(machineIdx, ms, self.parameters)
        new_time = timeTaken((os, newMs), self.parameters)
        if self.previous_time > new_time:
            return (os, newMs), new_time
        return (os, ms), self.previous_time

    # 9. 工序码随机反转子序列并同时随机改变对应位置机器码 已测
    def heuristic6(self):
        (os, ms) = self.previous_solution
        ida = idb = random.randint(0, len(os) - 2)
        while ida == idb:
            idb = random.randint(0, len(os) - 1)
        if ida > idb:
            ida, idb = idb, ida
        rev = os[ida:idb + 1]
        rev.reverse()
        newOs = os[:ida] + rev + os[idb + 1:]
        newMs = ms.copy()
        for i in range(ida, idb + 1):
            # print('place: ', i)
            newMs = changeMsRandom(i, newMs, self.parameters)
        new_time = timeTaken((newOs, newMs), self.parameters)
        if self.previous_time > new_time:
            return (newOs, newMs), new_time
        return (os, ms), self.previous_time



    # shaking=========================================================
    # 随机机器码区间破坏
    def heuristicA(self):
        (os, ms) = self.previous_solution
        ida = idb = random.randint(0, len(os) - 2)
        while ida == idb:
            idb = random.randint(0, len(os) - 1)
        if ida > idb:
            ida, idb = idb, ida
        newMs = ms.copy()
        for i in range(0, idb - ida + 1):
            newMs = changeMsRandom(i, newMs, self.parameters)
        new_time = timeTaken((os, newMs), self.parameters)
        return (os, newMs), new_time

    # 随机工序码区间破坏
    def heuristicB(self):
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
        new_time = timeTaken((newOs, ms), self.parameters)
        return (newOs, ms), new_time

    # 并行区间破坏
    def heuristicC(self):
        # 选择操作区间
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
        #随机改变区间工序码对应机器码
        newMs = ms.copy()
        for i in range(0, idb - ida + 1):
            machineIdx = getMachineIdx(i, newOs, self.parameters)
            newMs = changeMsRandom(machineIdx, newMs, self.parameters)
        # 替换当前解,跳出邻域
        new_time = timeTaken((newOs, newMs), self.parameters)
        return (newOs, newMs), new_time

    # 小邻域结构==============================

    # 8. 工序码随机交换同时随机改变对应位置机器码 已测
    def heuristicD(self):

        (os, ms) = self.previous_solution
        while(True):
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

            # 替换当前解,跳出邻域
            new_time = timeTaken((newOs, newMs), self.parameters)

            return (newOs, newMs), new_time

    # 5. 随机改变单个机器码 已测
    def heuristicE(self):
        (os, ms) = self.previous_solution
        while(True):
            machineIdx = random.randint(0, len(ms) - 1)
            # ('selected idx : ', machineIdx)
            newMs = changeMsRandom(machineIdx, ms, self.parameters)
            new_Time = timeTaken((os, newMs), self.parameters)
            # if new_Time != self.previous_time:
            #     # 替换当前解,跳出邻域
            #     self.previous_solution = (os, newMs)
            #     self.previous_time = new_Time
            #     return
            return (os, newMs), new_Time


    # 1. 随机交换两个工序码, 返回新的工序码 已测
    def heuristicF(self):
        # print('1')
        # 随机选择两个不同机器码
        (os, ms) = self.previous_solution
        while(True):
            ida = idb = random.randint(0, len(os) - 1)
            while ida == idb:
                idb = random.randint(0, len(os) - 1)

            newOs = os.copy()
            newOs[ida], newOs[idb] = newOs[idb], newOs[ida]
            new_time = timeTaken((newOs, ms), self.parameters)
            # if new_time != self.previous_time:
            #     # 替换当前解,跳出邻域
            #     self.previous_solution = (newOs, ms)
            #     self.previous_time = new_time
            #     return
            return (newOs, ms), new_time

    def heuristicG(self):
        # 选择操作区间
        (os, ms) = self.previous_solution
        ida = idb = random.randint(0, len(os) - 2)
        while ida == idb:
            idb = random.randint(0, len(os) - 1)
        if ida > idb:
            ida, idb = idb, ida

        newOs = os.copy()


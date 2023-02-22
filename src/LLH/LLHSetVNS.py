import random

from src.LLH.LLHUtils import timeTaken, getMachineIdx, changeMsRandom
from src.utils import encoding
from src.utils.parser import parse


class LLHSetVNS():
    def __init__(self, train = False):
        # solutions
        # self.solution_population = []
        # self.solution_population_time = []
        self.train = train
        # if self.train:
        #     self.best_solution = self.previous_solution = encoding.initializeFixedResult(self.parameters)
        # else:
        #     self.best_solution = self.previous_solution = encoding.initializeResult(self.parameters)
        # self.best_time = self.previous_time = timeTaken(self.previous_solution, self.parameters)
        # print('prev:', self.previous_time)


        #llh
        self.llh = []
        # 添加所有函数到 llh
        self.llh.append(self.heuristic1)
        self.llh.append(self.heuristic2)
        self.llh.append(self.heuristic3)
        self.llh.append(self.heuristic4)
        self.llh.append(self.heuristic5)
        # self.llh.append(self.heuristicA)
        self.llh.append(self.heuristicB)
        # self.llh.append(self.heuristicC)
        self.llh.append(self.heuristicD)
        self.llh.append(self.heuristicE)
        self.llh.append(self.heuristicF)

    # 工具方法,更新最优解
    def update_best_solution(self):
        if self.best_time > self.previous_time:
            self.best_solution = self.previous_solution
            self.best_time = self.previous_time


    # 重设
    def reset(self, problem_path):
        self.parameters = parse(problem_path)

        if self.train:
            self.best_solution = self.previous_solution = encoding.initializeFixedResult(self.parameters)
        else:
            self.best_solution = self.previous_solution = encoding.initializeResult(self.parameters)
        self.previous_time = self.best_time = timeTaken(self.previous_solution, self.parameters)



    # VND==============================================
    # 工序码局部搜索,返回值为 previous_time 改写完成
    def heuristic1(self):

        idx = random.randint(0, len(self.previous_solution[0]) - 1)

        for i in range(0, len(self.previous_solution[0])):
            newOs = self.previous_solution[0].copy()
            k = newOs[idx]
            newOs = newOs[0:idx] + newOs[idx + 1: len(newOs)]
            newOs = newOs[0: i] + [k] + newOs[i: len(newOs)]
            # 有优化即结束
            new_time = timeTaken((newOs, self.previous_solution[1]), self.parameters)
            if self.previous_time > new_time:
                self.previous_solution = (newOs, self.previous_solution[1])
                self.previous_time = new_time
                self.update_best_solution()
                return

    # 机器码局部搜索，全搜一遍,返回 previous_time 改写完成
    def heuristic2(self):
        # self.check_tabu(os_ms)
        # 获取作业集合
        jobs = self.parameters['jobs']
        # idx = self.get_randon_zero_index(self.ms_tabu)
        # idx = random.randint(0, len(ms) - 1)
        # self.ms_tabu[idx] = 1
        # 搜索整个机器码序列
        for idx in range(0, len(self.previous_solution[1])):
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
                newMs = self.previous_solution[1].copy()
                newMs[idx] = i
                new_time = timeTaken((self.previous_solution[0], newMs), self.parameters)
                if self.previous_time > new_time:
                    self.previous_solution = (self.previous_solution[0], newMs)
                    self.previous_time = new_time
                    break
        self.update_best_solution()

    # 并行局部搜索 改进在此
    def heuristic3(self):
        # self.check_tabu(os_ms)

        # 获取作业集合
        jobs = self.parameters['jobs']
        # print(tos)
        idx = random.randint(0, len(self.previous_solution[0]) - 1)
        # idx = self.get_randon_zero_index(self.os_tabu2)
        # self.os_tabu2[idx] = 1
        # 获取作业编号
        for i in range(0, len(self.previous_solution[0])):
            newOs = self.previous_solution[0].copy()
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
            opIdx = machineIdx - mcLength
            for j in range(0, len(jobs[jobIdx][opIdx])):
                newMs = self.previous_solution[1].copy()
                newMs[machineIdx] = j
                new_time = timeTaken((newOs, newMs), self.parameters)
                if self.previous_time > new_time:
                    self.previous_solution = (newOs, newMs)
                    self.previous_time = new_time
                    break
        self.update_best_solution()

    # 10. 随机前移工序码子序列, 并改变对应位置的机器码
    def heuristic4(self):
        ida = idb = random.randint(0, len(self.previous_solution[0]) - 2)
        while ida == idb:
            idb = random.randint(0, len(self.previous_solution[0]) - 1)
        if ida > idb:
            ida, idb = idb, ida
        newOs = self.previous_solution[0][ida:idb + 1] + self.previous_solution[0][:ida] + self.previous_solution[0][idb + 1:]
        newMs = self.previous_solution[1].copy()
        for i in range(0, idb - ida + 1):
            newMs = changeMsRandom(i, newMs, self.parameters)
        new_time = timeTaken((newOs, newMs), self.parameters)
        if self.previous_time > new_time:
            self.previous_solution = (newOs, newMs)
            self.previous_time = new_time
            self.update_best_solution()

    # 5. 随机改变单个机器码
    def heuristic5(self):
        machineIdx = random.randint(0, len(self.previous_solution[1]) - 1)
        # ('selected idx : ', machineIdx)
        newMs = changeMsRandom(machineIdx, self.previous_solution[1], self.parameters)
        new_time = timeTaken((self.previous_solution[0], newMs), self.parameters)
        if self.previous_time > new_time:
            self.previous_solution = (self.previous_solution[0], newMs)
            self.previous_time = new_time
            self.update_best_solution()

    # 9. 工序码随机反转子序列并同时随机改变对应位置机器码 已测
    def heuristic6(self):
        ida = idb = random.randint(0, len(self.previous_solution[0]) - 2)
        while ida == idb:
            idb = random.randint(0, len(self.previous_solution[0]) - 1)
        if ida > idb:
            ida, idb = idb, ida
        rev = self.previous_solution[0][ida:idb + 1]
        rev.reverse()
        newOs = self.previous_solution[0][:ida] + rev + self.previous_solution[0][idb + 1:]
        newMs = self.previous_solution[1].copy()
        for i in range(ida, idb + 1):
            # print('place: ', i)
            newMs = changeMsRandom(i, newMs, self.parameters)
        new_time = timeTaken((newOs, newMs), self.parameters)
        if self.previous_time > new_time:
            self.previous_solution = (newOs, newMs)
            self.previous_time = new_time
            self.update_best_solution()



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
        self.previous_solution = (os, newMs)
        self.previous_time = new_time

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
        self.previous_solution = (newOs, ms)
        self.previous_time = new_time

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
        self.previous_solution = (newOs, newMs)
        self.previous_time = new_time

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
            # if new_time != self.previous_time:
            #     self.previous_solution = (newOs, newMs)
            #     self.previous_time = new_time
            #     return
            self.previous_solution = (newOs, newMs)
            self.previous_time = new_time
            return

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
            self.previous_solution = (os, newMs)
            self.previous_time = new_Time
            return


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
            self.previous_solution = (newOs, ms)
            self.previous_time = new_time
            return

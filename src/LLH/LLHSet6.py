import random
import numpy as np

from src.LLH.LLHUtils import timeTaken, changeMsRandom, getMachineIdx
class LLHSet6:

    def __init__(self):
        #self.init_solution = init_solution
        # self.os_tabu = [0 for i in range(0, len(init_solution[0]))]
        # self.ms_tabu = [0 for i in range(0, len(init_solution[1]))]
        self.ms_tabu = None
        self.os_tabu = None
        self.os_tabu2 = None
        self.llh = []
        self.llh.append(self.heuristic1)
        self.llh.append(self.heuristic2)
        self.llh.append(self.heuristic3)
        self.llh.append(self.heuristic4)
        self.llh.append(self.heuristic5)
        self.llh.append(self.heuristic6)
        self.llh.append(self.heuristic7)
        self.llh.append(self.heuristic8)
        self.llh.append(self.heuristic9)
        self.llh.append(self.heuristic10)
        self.llh.append(self.heuristic11)

    def get_randon_zero_index(self, tabu_list):
        zero_indices = np.where(np.array(tabu_list) == 0)[0]
        return random.choice(zero_indices)


    # 破禁条件
    def check_tabu(self, solution):
        if self.os_tabu is None:
            self.os_tabu = [0 for i in range(0, len(solution[0]))]
        if self.os_tabu2 is None:
            self.os_tabu2 = [0 for i in range(0, len(solution[0]))]
        if self.ms_tabu is None:
            self.ms_tabu = [0 for i in range(0, len(solution[1]))]

        # 如果 os_tabu 各元素之和等于列表长度,将其全部赋值为 0
        if sum(self.os_tabu) == len(self.os_tabu):
            self.os_tabu = [0 for i in range(0, len(self.os_tabu))]
        # 如果 os_tabu2 各元素之和等于列表长度,将其全部赋值为 0
        if sum(self.os_tabu2) == len(self.os_tabu2):
            self.os_tabu2 = [0 for i in range(0, len(self.os_tabu2))]
        # 如果 ms_tabu 各元素之和等于列表长度,将其全部赋值为 0
        if sum(self.ms_tabu) == len(self.ms_tabu):
            self.ms_tabu = [0 for i in range(0, len(self.ms_tabu))]


    ##############优化操作#################
    # 1. 对 os 局部搜索
    def heuristic1(self, os_ms, parameters):

        self.check_tabu(os_ms)
        (os, ms) = os_ms
        # tos = os.copy()
        # print(tos)
        # idx = random.randint(0, len(tos) - 1)
        idx = self.get_randon_zero_index(self.os_tabu)
        self.os_tabu[idx] = 1

        bestTime = timeTaken((os, ms), parameters)
        # print('selected position: ', idx)
        for i in range(0, len(os)):
            newOs = os.copy()
            k = newOs[idx]
            newOs = newOs[0:idx] + newOs[idx + 1: len(newOs)]
            newOs = newOs[0: i] + [k] + newOs[i: len(newOs)]
            # print(newOs)
            if bestTime > timeTaken((newOs, ms), parameters):
                return (newOs, ms)
        return (os, ms)

    # 2. 机器码局部搜索，全搜一遍
    def heuristic2(self, os_ms, parameters):
        self.check_tabu(os_ms)
        (os, ms) = os_ms
        bestTime = timeTaken((os, ms), parameters)
        # 获取作业集合
        jobs = parameters['jobs']
        #print('ms_tabu', self.ms_tabu)
        idx = self.get_randon_zero_index(self.ms_tabu)
        self.ms_tabu[idx] = 1
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
                if bestTime > timeTaken((os, newMs), parameters):
                    return (os, newMs)
        return (os, ms)



    # 3. 并行局部搜索 改进在此
    def heuristic3(self, os_ms, parameters):
        #print('called')
        #print('called!')
        self.check_tabu(os_ms)
        (os, ms) = os_ms

        bestTime = timeTaken((os, ms), parameters)
        # 获取作业集合
        jobs = parameters['jobs']
        # print(tos)
        # idx = random.randint(0, len(tos) - 1)
        idx = self.get_randon_zero_index(self.os_tabu2)
        self.os_tabu2[idx] = 1

        # 获取作业编号

        #print('selected position: ', idx)

        for i in range(0, len(os)):
            newOs = os.copy()
            k = newOs[idx]
            newOs = newOs[0:idx] + newOs[idx + 1: len(newOs)]
            newOs = newOs[0: i] + [k] + newOs[i: len(newOs)]
            # 工序新位置到位
            # 开始机器码搜索
            machineIdx = getMachineIdx(i, newOs, parameters)
            #print('machineIdx: ', machineIdx)
            mcLength = 0  # 工具人
            jobIdx = -1  # 所属工作号
            for job in jobs:
                jobIdx += 1
                if mcLength + len(job) >= machineIdx + 1:
                    break
                else:
                    mcLength += len(job)
            #print('jobIdx: ', jobIdx)
            opIdx = machineIdx - mcLength
            #print('opIdx: ', opIdx)
            for j in range(0, len(jobs[jobIdx][opIdx])):
                newMs = ms.copy()
                newMs[machineIdx] = j
                if bestTime > timeTaken((newOs, newMs), parameters):
                    return (newOs, newMs)
        return (os, ms)

    # =====================变异操作++++++++++++++++++
    # 4 随机交换两个工序码, 返回新的工序码 已测
    def heuristic4(self, os_ms, parameters):
        # print('1')
        # 随机选择两个不同机器码
        (os, ms) = os_ms
        ida = idb = random.randint(0, len(os) - 1)
        while ida == idb:
            idb = random.randint(0, len(os) - 1)

        newOs = os.copy()
        newOs[ida], newOs[idb] = newOs[idb], newOs[ida]
        return (newOs, ms)

    # 5. 随机反转工序码子序列 已测
    def heuristic5(self, os_ms, parameters):
        (os, ms) = os_ms
        ida = idb = random.randint(0, len(os) - 2)
        while ida == idb:
            idb = random.randint(0, len(os) - 1)

        if ida > idb:
            ida, idb = idb, ida

        rev = os[ida:idb + 1]
        rev.reverse()
        newOs = os[:ida] + rev + os[idb + 1:]

        return (newOs, ms)

    # 6. 随机抽取一段长度为tos长度一半的子序列,将其顺序随机打乱
    def heuristic6(self, os_ms, parameters=None):
        (os, ms) = os_ms
        tos = os.copy()
        # 随机抽取一段tos的子序列,其长度为tos长度的一半
        ida = idb = random.randint(0, len(os) - 2)
        while ida == idb:
            idb = random.randint(0, len(os) - 1)

        if ida > idb:
            ida, idb = idb, ida
        # print('start: ', start, 'end: ', end)
        # print('tos: ', tos)
        # print(tos[0:start])

        mid = tos[ida:idb + 1]
        # print(mid)
        random.shuffle(mid)
        # print(mid)
        # print(tos[end:len(tos)])
        newOs = os[:ida] + mid + os[idb + 1:]
        return (newOs, ms)

    # 7. 随机改变单个机器码 已测
    def heuristic7(self, os_ms, parameters):
        (os, ms) = os_ms
        machineIdx = random.randint(0, len(ms) - 1)
        # ('selected idx : ', machineIdx)
        return (os, changeMsRandom(machineIdx, ms, parameters))

    # 8. 随机前移工序码子序列 已测
    def heuristic8(self, os_ms, parameters):
        (os, ms) = os_ms
        ida = idb = random.randint(0, len(os) - 2)
        while ida == idb:
            idb = random.randint(0, len(os) - 1)

        if ida > idb:
            ida, idb = idb, ida

        newOs = os[ida:idb + 1] + os[:ida] + os[idb + 1:]

        return (newOs, ms)

    # 9. 工序码随机交换同时随机改变对应位置机器码 已测
    def heuristic9(self, os_ms, parameters):
        jobs = parameters['jobs']
        (os, ms) = os_ms
        newOs = os.copy()
        newMs = ms.copy()

        ida = idb = random.randint(0, len(os) - 1)
        while ida == idb:
            idb = random.randint(0, len(os) - 1)

        newOs[ida], newOs[idb] = newOs[idb], newOs[ida]  # 工序码交换完成
        machineIda = getMachineIdx(ida, os, parameters)
        machineIdb = getMachineIdx(idb, os, parameters)

        newMs = changeMsRandom(machineIda, newMs, parameters)
        newMs = changeMsRandom(machineIdb, newMs, parameters)

        return (newOs, newMs)

    # 10. 工序码随机反转子序列并同时随机改变对应位置机器码 已测
    def heuristic10(self, os_ms, parameters):
        (os, ms) = os_ms
        ida = idb = random.randint(0, len(os) - 2)
        while ida == idb:
            idb = random.randint(0, len(os) - 1)

        if ida > idb:
            ida, idb = idb, ida

        # print('start: ', ida, ' end: ', idb)

        rev = os[ida:idb + 1]
        rev.reverse()
        newOs = os[:ida] + rev + os[idb + 1:]
        newMs = ms.copy()
        for i in range(ida, idb + 1):
            # print('place: ', i)
            newMs = changeMsRandom(i, newMs, parameters)

        return (newOs, newMs)

    # 11. 随机前移工序码子序列, 并改变对应位置的机器码 已测
    def heuristic11(self, os_ms, parameters):
        (os, ms) = os_ms
        ida = idb = random.randint(0, len(os) - 2)
        while ida == idb:
            idb = random.randint(0, len(os) - 1)

        if ida > idb:
            ida, idb = idb, ida

        newOs = os[ida:idb + 1] + os[:ida] + os[idb + 1:]
        newMs = ms.copy()
        for i in range(0, idb - ida + 1):
            newMs = changeMsRandom(i, newMs, parameters)

        return (newOs, newMs)
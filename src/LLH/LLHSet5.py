import random
from src.LLH.LLHUtils import timeTaken, changeMsRandom, getMachineIdx

class LLHSet5:
    def __init__(self):
        self.llh = []
        #添加所有函数到 llh
        self.llh.append(self.heuristic1)
        self.llh.append(self.heuristic2)
        self.llh.append(self.heuristic3)
        self.llh.append(self.heuristic4)
        self.llh.append(self.heuristic5)
        self.llh.append(self.heuristic6)

    # =====================启发式操作++++++++++++++++++

    # 4. 对 os 简化领域搜索 已测
    def heuristic1(self, os_ms, parameters):
        (os, ms) = os_ms
        tos = os.copy()
        # print(tos)
        idx = random.randint(0, len(tos) - 1)
        bestTime = timeTaken((tos, ms), parameters)
        # print('selected position: ', idx)
        for i in range(0, len(tos)):
            newOs = tos.copy()
            k = newOs[idx]
            newOs = newOs[0:idx] + newOs[idx + 1: len(newOs)]
            newOs = newOs[0: i] + [k] + newOs[i: len(newOs)]
            # print(newOs)
            if bestTime > timeTaken((newOs, ms), parameters):
                tos = newOs
        return (tos, ms)


    # 6. 机器码简化领域搜索 已测
    def heuristic2(self, os_ms, parameters):
        (os, ms) = os_ms
        tms = ms.copy()
        bestTime = timeTaken((os, tms), parameters)
        for i in range(0, len(tms)):
            newMs = changeMsRandom(i, ms, parameters)
            if bestTime > timeTaken((os, newMs), parameters):
                tms = newMs

        return (os, tms)

    # 7. 并行简化领域搜索
    def heuristic3(self, os_ms, parameters):
        (os, ms) = os_ms
        tos = os.copy()
        tms = ms.copy()
        # print(tos)
        idx = random.randint(0, len(tos) - 1)

        bestTime = timeTaken((tos, ms), parameters)
        # print('selected position: ', idx)
        for i in range(0, len(tos)):
            newOs = tos.copy()
            k = newOs[idx]
            newOs = newOs[0:idx] + newOs[idx + 1: len(newOs)]
            newOs = newOs[0: i] + [k] + newOs[i: len(newOs)]
            machineIdx = getMachineIdx(i, os, parameters)
            newMs = changeMsRandom(machineIdx, ms, parameters)
            # print(newOs)
            if bestTime > timeTaken((newOs, newMs), parameters):
                tos = newOs
                tms = newMs
        return (tos, tms)

###############变异
    # 5. 随机反转工序码子序列 已测
    def heuristic4(self, os_ms, parameters):
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
    def heuristic5(self, os_ms, parameters=None):
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

    # 5. 随机改变单个机器码 已测
    def heuristic6(self, os_ms, parameters):
        (os, ms) = os_ms
        machineIdx = random.randint(0, len(ms) - 1)
        # ('selected idx : ', machineIdx)
        return (os, changeMsRandom(machineIdx, ms, parameters))
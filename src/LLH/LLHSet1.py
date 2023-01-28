import random
from src.LLH.LLHUtils import timeTaken, changeMsRandom, getMachineIdx


# =====================启发式操作++++++++++++++++++
# 1. 随机交换两个工序码, 返回新的工序码 已测
def heuristic1(os_ms, parameters):
    # print('1')
    # 随机选择两个不同机器码
    (os, ms) = os_ms
    ida = idb = random.randint(0, len(os) - 1)
    while ida == idb:
        idb = random.randint(0, len(os) - 1)

    newOs = os.copy()
    newOs[ida], newOs[idb] = newOs[idb], newOs[ida]
    return (newOs, ms)


# 2. 随机反转工序码子序列 已测
def heuristic2(os_ms, parameters):
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


# 3. 随机前移工序码子序列 已测
def heuristic3(os_ms, parameters):
    (os, ms) = os_ms
    ida = idb = random.randint(0, len(os) - 2)
    while ida == idb:
        idb = random.randint(0, len(os) - 1)

    if ida > idb:
        ida, idb = idb, ida

    newOs = os[ida:idb + 1] + os[:ida] + os[idb + 1:]

    return (newOs, ms)


# 4. 对 os 简化领域搜索 已测
def heuristic4(os_ms, parameters):
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


# 5. 随机改变单个机器码 已测
def heuristic5(os_ms, parameters):
    (os, ms) = os_ms
    machineIdx = random.randint(0, len(ms) - 1)
    # ('selected idx : ', machineIdx)
    return (os, changeMsRandom(machineIdx, ms, parameters))


# 6. 机器码简化领域搜索 已测
def heuristic6(os_ms, parameters):
    (os, ms) = os_ms
    tms = ms.copy()
    bestTime = timeTaken((os, tms), parameters)
    for i in range(0, len(tms)):
        newMs = changeMsRandom(i, ms, parameters)
        if bestTime > timeTaken((os, newMs), parameters):
            tms = newMs

    return (os, tms)


# 7. 并行简化领域搜索
def heuristic7(os_ms, parameters):
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


# 8. 工序码随机交换同时随机改变对应位置机器码 已测
def heuristic8(os_ms, parameters):
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


# 9. 工序码随机反转子序列并同时随机改变对应位置机器码 已测
def heuristic9(os_ms, parameters):
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


# 10. 随机前移工序码子序列, 并改变对应位置的机器码 已测
def heuristic10(os_ms, parameters):
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

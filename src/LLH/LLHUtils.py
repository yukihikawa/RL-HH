import math
import random


from src.utils import decoding

# 返回一个包含所有 LLH 的 list, 便于调用


# 以模拟退火机制, 调用 function,参数为 solution 和 parameters.
# 如果 solution 的 timeTaken(solution, parameters) 比原来的小,则接受新的 solution
# 否则以一定概率接受新的 solution,并存储历史上的最优值
# 降温过程结束后返回历史最优解
def SAWarapper(function, solution, parameters, maxTemp=100, minTemp=0.01, coolingRate=0.9):
    bestSolution = solution
    bestTime = timeTaken(bestSolution, parameters)
    currentSolution = solution
    currentTemp = maxTemp
    while currentTemp > minTemp:
        newSolution = function(currentSolution, parameters)
        newTime = timeTaken(newSolution, parameters)
        if newTime < bestTime:
            bestSolution = newSolution
            bestTime = newTime
        else:
            p = math.exp(-(newTime - bestTime) / currentTemp)
            if random.random() < p:
                currentSolution = newSolution
        currentTemp *= coolingRate
    return bestSolution

#
def SAWarapper2(function, solution, parameters, maxTemp=100, minTemp=0.01, coolingRate=0.9):
    bestSolution = solution
    bestTime = timeTaken(bestSolution, parameters)
    currentSolution = solution
    currentTemp = maxTemp
    FLAG = 0
    while currentTemp > minTemp:
        newSolution = function(currentSolution, parameters)
        newTime = timeTaken(newSolution, parameters)
        if newTime < bestTime:
            bestSolution = newSolution
            bestTime = newTime
            FLAG = 0
        else:
            p = math.exp(-(newTime - bestTime) / FLAG)
            if random.random() < p:
                currentSolution = newSolution
                FLAG = 0
            else:
                FLAG += 1
        currentTemp *= coolingRate
    return bestSolution

# ================== 工具方法 ===================
# 最大完成时间
def timeTaken(os_ms, pb_instance):
    (os, ms) = os_ms  # 元组
    decoded = decoding.decode(pb_instance, os, ms)  # 结构化的问题数据集

    # 每台机器的最大值
    max_per_machine = []
    for machine in decoded:
        max_d = 0
        for job in machine:  # 遍历机器的所有作业
            end = job[3] + job[1]
            if end > max_d:
                max_d = end
        max_per_machine.append(max_d)

    return max(max_per_machine)

# 使用已传入的解码数据计算时间
def timeTakenForDecoded(decoded_data):
    decoded = decoded_data  # 结构化的问题数据集
    # 每台机器的最大值
    max_per_machine = []
    for machine in decoded:
        max_d = 0
        for job in machine:  # 遍历机器的所有作业
            end = job[3] + job[1]
            if end > max_d:
                max_d = end
        max_per_machine.append(max_d)

    return max(max_per_machine)

# 使用传入的解码数据计算工作负载
def get_machine_workload(pb_instance, decoded):
    # (os, ms) = os_ms  # 元组
    # decoded = decoding.decode(pb_instance, os, ms)  # 结构化的问题数据集
    total_time = timeTakenForDecoded(decoded)
    # 每台机器的负载
    workload_per_machine = [0] * pb_instance['machinesNb']

    for i in range(len(decoded)):
        time = 0
        # 计算该机器的工作总量
        for job in decoded[i]:
            # print('job', job[0], ' : ', job[1])
            time += job[1]
        workload_per_machine[i] = time / total_time
    return workload_per_machine

# 改变指定机器码序列位置的机器码 已测
def changeMsRandom(machineIdx, ms, parameters):
    jobs = parameters['jobs']  # 作业的集合
    # jobIdx = os[machineIdx] # 指定的位置所属的作业序号 错误在此
    mcLength = 0  # 工具人
    jobIdx = -1  # 所属工作号

    for job in jobs:
        jobIdx += 1

        if mcLength + len(job) >= machineIdx + 1:
            break
        else:
            mcLength += len(job)

    opIdx = machineIdx - mcLength  # 指定位置对应的 在工件中的工序号

    # print('belongs to: job', jobIdx, ' op: ', opIdx, ' ava machine: ', len(jobs[jobIdx][opIdx]))
    newMachine = random.randint(0, len(jobs[jobIdx][opIdx]) - 1)
    newMs = ms.copy()
    newMs[machineIdx] = newMachine
    return newMs

# 获取指定 os 位置工序在 ms 中的位置
def getMachineIdx(jobIdx, os, parameters):
    jobNum = os[jobIdx]  # 工件号
    jobs = parameters['jobs']  # 工件集合
    machineIdx = 0  # 在 ms 中的位置
    for i in range(0, jobNum):  #
        machineIdx += len(jobs[i])
    for i in range(0, jobIdx):
        if os[i] == jobNum:
            machineIdx += 1
    return machineIdx

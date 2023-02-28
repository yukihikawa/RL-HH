import random

from src.LLH.LLHUtils import timeTaken, get_machine_workload, timeTakenForDecoded
from src.LLH.LLHolder import LLHolder
import os

from src.utils import decoding, gantt
from src.utils.decoding import split_ms, decode
from src.utils.encoding import initializeResult
from src.utils.parser import parse

def vnd13(parameters, current_solution):
    # 对当前解进行解码
    machine_operation = decode(parameters, current_solution[0], current_solution[1])
    # 获取工作负载
    workload = get_machine_workload(parameters, machine_operation)
    # 取得最大负载机器,workload中最大值的索引, 从 0 开始的
    max_workload_machine = workload.index(max(workload))
    # 从具有最大负载的机器中随机选择一个工序
    selected_op = random.choice(machine_operation[max_workload_machine])
    # 获取工序信息
    job_idx, op_idx = map(int, selected_op[0].split('-'))
    op_idx -= 1
    # 获取工序的机器集合
    machine_set = parameters['jobs'][job_idx][op_idx]
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
            print("sdfgsd:", selected_new_machine)
    print('selected_new_machine ggggg: ', selected_new_machine)
    # 生成新的ms编码
    ms_s = split_ms(parameters, current_solution[1])  # 分离的ms编码
    # 在 ms 中的位置
    ms_idx = 0
    for i in range(job_idx):
        ms_idx += len(ms_s[i])
    ms_idx += op_idx
    new_ms = current_solution[1].copy()
    # print('old ms: ', new_ms[ms_idx])
    new_ms[ms_idx] = selected_new_machine
    return (current_solution[0], new_ms)

problem = 'MK01'
problem_str = os.path.join(os.getcwd(), "../../Brandimarte_Data/" + problem + ".fjs")
holder = LLHolder(7)
llh = holder.set.llh[2]
llh2 = holder.set.llh[1]

parameters = parse(problem_str)
jobs = parameters['jobs']
for i in range(len(jobs)):
    print('job: ', i)
    for op in jobs[i]:
        print(op)

solution = initializeResult(parameters)
ms_s = split_ms(parameters, solution[1])
for i in range(len(ms_s)):
    print('job: ', i)
    for op in range(len(ms_s[i])):
        print('op: ', op+1, 'selected machine: ', jobs[i][op][ms_s[i][op]])

# print(solution)
decoded = decoding.decode(parameters, solution[0], solution[1])

for i in range(len(decoded)):
    print('machine: ', i)
    for op in decoded[i]:
        print(op)
print('================================================================')
print('prevTime:', timeTakenForDecoded(decoded))

workload = get_machine_workload(parameters, decoded)
print("workload", workload)
# 取得最大负载机器,workload中最大值的索引, 从 0 开始的
max_workload_machine = workload.index(max(workload))
print("max_workload_machine", max_workload_machine)
# 从具有最大负载的机器中随机选择一个工序
selected_op = random.choice(decoded[max_workload_machine])
print("selected_op", selected_op)
job_idx, op_idx = map(int, selected_op[0].split('-'))
op_idx -= 1
print("job_idx, op_idx", job_idx, op_idx)
machine_set = parameters['jobs'][job_idx][op_idx]
print("machine_set", machine_set)
# 当前工序所在机器负载
prev_load = max(workload)
# 从机器集合中选择负载最小的机器
print('prev_load', prev_load)
selected_new_machine = 0
for i in range(len(machine_set)): # 遍历机器合集
    machine_idx = machine_set[i]['machine']
    new_load = workload[machine_idx - 1]
    print('new_load', new_load)
    if new_load < prev_load:
        prev_load = new_load
        selected_new_machine = i
        break
print('selected_new_machine', selected_new_machine)

ms_s = split_ms(parameters, solution[1]) # 分离的ms编码
print("ms_s")
for i in range(len(ms_s)):
    print('job: ', i, ' ', ms_s[i])
# 在 ms 中的位置
ms_idx = 0
for i in range(job_idx):
    ms_idx += len(ms_s[i])
ms_idx += op_idx
new_ms = solution[1].copy()
new_ms[ms_idx] = selected_new_machine
ms_s2 = split_ms(parameters, new_ms) # 分离的ms编码
print("ms_s2")
for i in range(len(ms_s2)):
    print('job: ', i, ' ', ms_s2[i])
new_solution = (solution[0], new_ms)
new_decoded = decoding.decode(parameters, new_solution[0], new_solution[1])
workload = get_machine_workload(parameters, new_decoded)
print("workload", workload)
print('newTime:', timeTaken(new_solution, parameters))
# for i in range(10):
#     solution = vnd13(parameters, new_solution)
#     print('newTime:', timeTaken(new_solution, parameters))

# print(get_machine_workload(parameters, decoded))
# prevTime = timeTaken(solution, parameters)
# gantt_data = decoding.translate_decoded_to_gantt(decoding.decode(parameters, solution[0], solution[1]))
# gantt.draw_chart(gantt_data)
# for i in range(1000):
#     newSolution = llh(solution, parameters)
#     newSolution = llh2(newSolution, parameters)
#     print("new result",timeTaken(newSolution, parameters))
#     solution = newSolution
#     # print(solution)


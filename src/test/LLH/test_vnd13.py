import os
import random

from src.LLH.LLHUtils import get_machine_workload, timeTakenForDecoded, timeTaken
from src.utils import decoding
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
    #         print("sdfgsd:", selected_new_machine)
    # print('selected_new_machine ggggg: ', selected_new_machine)
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


problem = 'MK02'
problem_str = os.path.join(os.getcwd(), "../../Brandimarte_Data/" + problem + ".fjs")

parameters = parse(problem_str)
solution = initializeResult(parameters)
for i in range(10):
    decoded = decoding.decode(parameters, solution[0], solution[1])
    print('================================================================')
    print('prevTime:', timeTakenForDecoded(decoded))
    workload = get_machine_workload(parameters, decoded)
    print("workload", workload)
    solution = vnd13(parameters, solution)
    new_decoded = decoding.decode(parameters, solution[0], solution[1])
    workload = get_machine_workload(parameters, new_decoded)
    print("workload", workload)
    print('newTime:', timeTaken(solution, parameters))
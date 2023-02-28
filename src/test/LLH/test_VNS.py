import random

from src.LLH.LLHSetVNS import LLHSetVNS
import os

from src.LLH.LLHUtils import get_machine_workload
from src.utils.decoding import decode, split_ms


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
        new_load = workload[machine_idx]
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

problem = 'MK09'
problem_str = os.path.join(os.getcwd(), "../../Brandimarte_Data/" + problem + ".fjs")
set = LLHSetVNS()
# print(set.previous_time)
set.reset(problem_str)
# for i in range(0, 50):
#     # set.reset()
#     print('ori: ', set.previous_time)
#     for i in range(0, 100):
#         # 随机选取 0-4
#         idx = random.randint(0, 4)
#         # 执行对应的函数
#         set.llh[idx]()
#     print("local optimum: ", set.previous_time)
#     print('best: ', set.best_time)
#     # 随机选取 5-7
#     idx = random.randint(5, 7)
#     # 执行对应的函数
#     set.llh[idx]()
#     print("new: ", set.previous_time)
for i in range(0, 40):
    set.reset(problem_str)
    print('ori: ', set.previous_time)
    set.heuristicD()
    print("new: ", set.previous_time)
    print(' ')

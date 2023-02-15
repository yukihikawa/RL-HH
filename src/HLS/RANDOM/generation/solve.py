import math
import os
import random
import time

from src.LLH.LLHUtils import timeTaken
from src.LLH.LLHolder import LLHolder
from src.utils import encoding
from src.utils.parser import parse

PROBLEM = 'MK06'
ALG = 'LLH_SET2_LENGTH5000'
TIMES = 20
LLH_SET = 2

def run(alg_name, problem):
    not_accepted = 1
    t0 = time.time()
    #从 {alg_name}.txt文件中读取一个 list
    with open(f'{alg_name}.txt', 'r') as f:
        alg_list = eval(f.read())
    problem_path = os.path.join(os.getcwd(), "../../../Brandimarte_Data/" + problem + ".fjs")
    parameters = parse(problem_path)
    best_solution = prev_solution = (encoding.generateOS(parameters), encoding.generateMS(parameters))
    best_time = prev_time = timeTaken(best_solution, parameters)
    print('initial time: ', best_time)
    holder = LLHolder(LLH_SET)
    for i in alg_list:
        new_solution = holder.set.llh[i](prev_solution, parameters)
        new_time = timeTaken(new_solution, parameters)
        if new_time < prev_time:
            not_accepted = 1
            prev_solution = new_solution
            prev_time = new_time
            if new_time < best_time:
                best_time = new_time
                best_solution = new_solution
        else:
            if random.random() < math.exp(-(new_time - prev_time) / (not_accepted * 0.01)):
                prev_solution = new_solution
                prev_time = new_time
                not_accepted = 1
            else:
                not_accepted += 1
        #print('new best time: ', best_time)
    best_time = timeTaken(best_solution, parameters)
    print('final best time: ', best_time)
    t1 = time.time()
    print('time used: ', t1 - t0)
    return best_time, t1 - t0, best_solution

def test(alg_name, problem, times):
    result = []
    timeTaken = []
    for i in range(times):
        time, timeUsed, solution = run(alg_name, problem)
        result.append(time)
        timeTaken.append(timeUsed)
    print('alg_name: ', alg_name, 'problem: ', problem, 'LLH Set: ', LLH_SET)
    print('result: ', result)
    print('average time: ', sum(result) / times)
    print('timeTaken: ', timeTaken)
    print('average timeTaken: ', sum(timeTaken) / times)


if __name__ == '__main__':
    test(ALG, PROBLEM, TIMES)
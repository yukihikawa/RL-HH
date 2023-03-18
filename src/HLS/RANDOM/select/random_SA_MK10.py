import math
import random
import time
from concurrent.futures import ThreadPoolExecutor

import src.HLS.GEN.genetic.gen_main as gen_main
from src.HLS.GEN.genetic import gen_ops
from src.HLS.GEN.genetic.applyPopParallel import apply
from src.HLS.GEN.genetic.config import *
from src.LLH.LLHUtils import timeTaken
from src.LLH.LLHolder import LLHolder
from src.utils.encoding import initializeResult
from src.utils.parser import parse

PROBLEM = 'MK10'
GEN_NUM =9000
TEST_ITER = 20
LLH_SET = 1

def runForTest(problem, genNum, LLH):
    t0 = time.time()
    notAccepted = 1
    parameters = parse(problem)
    prevSolution = best_solution = initializeResult(parameters)
    prevTime = best_time = timeTaken(best_solution, parameters)
    llh_called = [0 for _ in range(len(LLH))]
    for gen in range(genNum):
        idx = random.randint(0, len(LLH) - 1)
        #print('gen:', gen, 'llh: ', idx, 'best time: ', best_time)
        llh_called[idx] += 1
        new_solution = LLH[idx](prevSolution, parameters)
        new_time = timeTaken(new_solution, parameters)
        if new_time < prevTime:
            notAccepted = 1
            prevTime = new_time
            prevSolution = new_solution
            if new_time < best_time:
                best_time = new_time
                best_solution = new_solution
                print('iter:', gen, 'new best time: ', best_time)
        else:
            if random.random() < math.exp(-(new_time - prevTime) / (notAccepted * 0.01)):
                prevSolution = new_solution
                prevTime = new_time
                notAccepted = 1
            else:
                notAccepted += 1

    t1 = time.time()
    print('time: ', t1 - t0)
    print('llh called:')
    print(llh_called)

    return best_time, t1 - t0

def test(TEST_ITER, PROBLEM, genNum, llh_set):
    problem_path = os.path.join(os.getcwd(), "../../../Brandimarte_Data/" + PROBLEM + ".fjs")
    LLH = LLHolder(LLH_SET).set.llh
    result = {}
    timeUsed = {}
    for i in range(TEST_ITER):
        bt, tt = runForTest(problem_path, genNum, LLH)
        result[i] = bt
        timeUsed[i] = tt
        print(result[i], timeUsed[i])
    print('problem: ', PROBLEM, 'genNum: ', genNum, 'LLH Set: ', llh_set)
    print(result)
    print(timeUsed)
    print('average time: ', sum(result.values()) / TEST_ITER)

if __name__ == '__main__':
    test(TEST_ITER, PROBLEM, GEN_NUM, LLH_SET)
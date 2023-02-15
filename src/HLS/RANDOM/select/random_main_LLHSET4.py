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
from src.LLH.LLHSet4 import LLHSet4

PROBLEM = 'MK02'
GEN_NUM =5000
TEST_ITER = 10
LLH_SET = 4

def runForTest(problem, genNum, LLH_SET):
    t0 = time.time()
    parameters = parse(problem)
    best_solution = initializeResult(parameters)
    best_time = timeTaken(best_solution, parameters)
    holder = LLHolder(LLH_SET)
    LLH = holder.set.llh
    llh_called = [0 for _ in range(len(LLH))]
    for gen in range(genNum):
        idx = random.randint(0, len(LLH) - 1)
        #print('gen:', gen, 'llh: ', idx, 'best time: ', best_time)
        llh_called[idx] += 1
        new_solution = LLH[idx](best_solution, parameters)
        new_time = timeTaken(new_solution, parameters)
        if new_time < best_time:
            best_time = new_time
            best_solution = new_solution
            #print('iter:', gen, 'new best time: ', best_time)
    t1 = time.time()
    print('time: ', t1 - t0)
    print('llh called:')
    print(llh_called)

    return best_time, t1 - t0

def test(TEST_ITER, PROBLEM, genNum, llh_set):
    problem_path = os.path.join(os.getcwd(), "../../../Brandimarte_Data/" + PROBLEM + ".fjs")

    result = {}
    timeUsed = {}
    for i in range(TEST_ITER):
        bt, tt = runForTest(problem_path, genNum, llh_set)
        result[i] = bt
        timeUsed[i] = tt
        print(result[i], timeUsed[i])
    print('problem: ', PROBLEM, 'genNum: ', genNum, 'LLH Set: ', llh_set)
    print(result)
    print(timeUsed)
    print('average time: ', sum(result.values()) / TEST_ITER)

if __name__ == '__main__':
    test(TEST_ITER, PROBLEM, GEN_NUM, LLH_SET)
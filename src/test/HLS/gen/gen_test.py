import src.HLS.GEN.genetic.gen_main as gen_main
from src.HLS.GEN.genetic import gen_ops
from src.HLS.GEN.genetic.config import *
from src.LLH.LLHUtils import timeTaken
from src.LLH.LLHolder import LLHolder
from src.utils.encoding import initializeResult
from src.utils.parser import parse


def test(TEST_ITER, PROBLEM, genNum, chromLength, popSize, crossTimes, pMut, llh_set):
    problem_path = os.path.join(os.getcwd(), "../../../Brandimarte_Data/" + PROBLEM + ".fjs")
    LLH = LLHolder(LLH_SET)
    result = {}
    for i in range(TEST_ITER):
        result[i] = runForTest(problem_path, genNum, chromLength, popSize, crossTimes, pMut, LLH)
        print(result[i])
    print('genNum: ', genNum, 'chromLength: ', chromLength, 'popSize: ', popSize, 'cross times: ', crossTimes, 'pM: ', pMut, 'LLH Set: ', llh_set)
    print('result for ', PROBLEM, ':')
    print(result)
    print('average time: ', sum(result.values()) / TEST_ITER)


def runForTest(problem, genNum, chromLength, popSize, crossTimes, pMut, LLH):
    parameters = parse(problem)
    best_solution = initializeResult(parameters)
    best_time = timeTaken(best_solution, parameters)
    population = gen_ops.initPopulation(popSize, chromLength, len(LLH))
    for gen in range(genNum):
        if gen % 5 == 0:
            print('generation: ', gen, 'best time: ', best_time)
        parentIdx1, parentIdx2 = gen_ops.randomSelection(population)
        chrom1, chrom2 = population[parentIdx1], population[parentIdx2]
        childPop = gen_ops.crossover(chrom1, chrom2, crossTimes)
        childPop = gen_ops.mutate(childPop, pMut, len(LLH))
        new_solutions, new_times = gen_ops.applyPopulation(childPop, best_solution, parameters, LLH)
        fitness = gen_ops.fitness(new_times)
        for i in range(len(new_solutions)):
            new_time = new_times[i]
            if new_time < best_time:
                best_time = new_time
                best_solution = new_solutions[i]
        childChrom1 = gen_ops.bestSelection(childPop, fitness)
        childChrom2 = gen_ops.rouletteSelection(childPop, fitness)
        population[parentIdx1], population[parentIdx2] = childChrom1, childChrom2
    return best_time









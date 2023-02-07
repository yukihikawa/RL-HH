import src.HLS.GEN.genetic.gen_main as gen_main
from src.HLS.GEN.genetic import gen_ops
from src.HLS.GEN.genetic.config import *
from src.LLH.LLHUtils import timeTaken
from src.utils.encoding import initializeResult
from src.utils.parser import parse


def test(TEST_ITER, PROBLEM, genNum, chromLength, popSize, crossTimes, pMut):
    problem_path = os.path.join(os.getcwd(), "../../../Brandimarte_Data/" + PROBLEM + ".fjs")
    print('result for ', PROBLEM, ':')
    result = {}
    for i in range(TEST_ITER):
        result[i] = runForTest(problem_path, genNum, chromLength, popSize, crossTimes, pMut)
        print(result[i])
    print(result)

def runForTest(problem, genNum, chromLength, popSize, crossTimes, pMut):
    parameters = parse(problem)
    best_solution = initializeResult(parameters)
    best_time = timeTaken(best_solution, parameters)
    population = gen_ops.initPopulation(popSize, chromLength)
    for gen in range(genNum):
        if gen % 5 == 0:
            print('generation: ', gen, 'best time: ', best_time)
        parentIdx1, parentIdx2 = gen_ops.randomSelection(population)
        chrom1, chrom2 = population[parentIdx1], population[parentIdx2]
        childPop = gen_ops.crossover(chrom1, chrom2, crossTimes)
        childPop = gen_ops.mutate(childPop, pMut)
        new_solutions, new_times = gen_ops.applyPopulation(childPop, best_solution, parameters)
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

if __name__ == '__main__':
    for i in range(6, 10):
        test(PROBLEM_SET[i])







import random

from src.LLH.LLHolder import LLHolder
from src.utils.encoding import initializeResult
from src.LLH.LLHUtils import timeTaken
from src.utils.parser import parse
import src.HLS.GEN.genetic.gen_ops as gen_ops
import src.HLS.GEN.genetic.config as config
from src.HLS.GEN.genetic.applyPopParallel import apply

# 运行遗传算法
def run(problem):
    print(config.PROBLEM)
    print('gen:', config.GEN_NUM, 'chrom: ', config.CHROM_LENGTH, 'llh: ', config.LLH_SET)
    parameters = parse(problem)
    best_solution = initializeResult(parameters)
    best_time = timeTaken(best_solution, parameters)
    LLH = LLHolder(config.LLH_SET)
    #初始化种群
    population = gen_ops.initPopulation(config.POP_SIZE, config.CHROM_LENGTH, len(LLH))
    applier = apply(best_solution, parameters, LLH)
    #开始循环
    for gen in range(config.GEN_NUM):
        print('generation: ', gen, 'best time: ', best_time)
        # 随机选出两个父代
        parentIdx1, parentIdx2 = gen_ops.randomSelection(population)
        chrom1, chrom2 = population[parentIdx1], population[parentIdx2]
        # 交叉生成新子代种群
        childPop = gen_ops.crossover(chrom1, chrom2, config.CROSS_TIMES)
        # 子代种群变异
        childPop = gen_ops.mutate(childPop, config.P_MUT, len(LLH))
        # 将子代种群应用到 best_solution, 获取所有新解
        applier.solution = best_solution
        new_solutions = applier.applyPopulation(childPop)
        new_times = []
        fitness = []
        for solution in new_solutions:
            newTime = timeTaken(solution, parameters)
            new_times.append(newTime)
            fitness.append(1 / newTime)
        # 获取 new_times中最小值的索引
        idx = new_times.index(min(new_times))
        #if new_times[idx] < best_time:
        best_time = new_times[idx]
        best_solution = new_solutions[idx]
        print('new best time: ', best_time)


        #子代选择两个个体
        childChrom1 = gen_ops.bestSelection(childPop, fitness)
        childChrom2 = gen_ops.rouletteSelection(childPop, fitness)
        #替换所选的父代个体
        population[parentIdx1], population[parentIdx2] = childChrom1, childChrom2
    print('final best time: ', best_time)
    return best_time



if __name__ == '__main__':
    run(config.PROBLEM_PATH)
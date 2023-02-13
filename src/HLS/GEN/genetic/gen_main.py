import random
from concurrent.futures import ThreadPoolExecutor

from src.LLH.LLHolder import LLHolder
from src.utils.encoding import initializeResult
from src.LLH.LLHUtils import timeTaken
from src.utils.parser import parse
import src.HLS.GEN.genetic.gen_ops as gen_ops
import src.HLS.GEN.genetic.config as config


# 运行遗传算法
def run(problem):

    print(config.PROBLEM)
    print('gen:', config.GEN_NUM, 'chrom: ', config.CHROM_LENGTH, 'llh: ', config.LLH_SET)
    parameters = parse(problem)
    best_solution = initializeResult(parameters)
    best_time = timeTaken(best_solution, parameters)
    holder = LLHolder(config.LLH_SET)
    LLH = holder.set.llh
    #初始化种群
    population = gen_ops.initPopulation(config.POP_SIZE, config.CHROM_LENGTH, len(LLH))
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
        new_solutions, new_times = gen_ops.applyPopulation(childPop, best_solution, parameters, LLH)
        # 计算子代适应度
        fitness = gen_ops.fitness(new_times)
        # 获取 new_times中最小值的索引
        idx = new_times.index(min(new_times))
        #if new_times[idx] < best_time:
        best_time = new_times[idx]
        best_solution = new_solutions[idx]
        print('new best time: ', best_time)
        # for i in range(len(new_solutions)):
        #     new_time = new_times[i]
        #     if new_time < best_time:
        #         best_time = new_time
        #         best_solution = new_solutions[i]
        #         print('new best time: ', best_time)
            # elif new_time == best_time:
            #     p = random.random()
            #     if p < 0.6 - 0.005 * gen:
            #         best_time = new_time
            #         best_solution = new_solutions[i]
            #         print('new best time: ', best_time)

        #子代选择两个个体
        childChrom1 = gen_ops.bestSelection(childPop, fitness)
        childChrom2 = gen_ops.rouletteSelection(childPop, fitness)
        #替换所选的父代个体
        population[parentIdx1], population[parentIdx2] = childChrom1, childChrom2
    print('final best time: ', best_time)
    return best_time



if __name__ == '__main__':
    run(config.PROBLEM_PATH)
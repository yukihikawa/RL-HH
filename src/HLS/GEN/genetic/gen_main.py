from src.utils.encoding import initializeResult
from src.LLH.LLHUtils import timeTaken
from src.utils.parser import parse
import src.HLS.GEN.genetic.gen_ops as gen_ops
import src.HLS.GEN.genetic.config as config

# 运行遗传算法
def run(problem):
    parameters = parse(problem)
    best_solution = initializeResult(parameters)
    best_time = timeTaken(best_solution, parameters)
    #初始化种群
    population = gen_ops.initPopulation(config.POP_SIZE, config.CHROM_LENGTH)
    #开始循环
    for i in range(config.GEN_NUM):
        print('generation: ', i, 'best time: ', best_time)
        # 随机选出两个父代
        parentIdx1, parentIdx2 = gen_ops.randomSelection(population)
        chrom1, chrom2 = population[parentIdx1], population[parentIdx2]
        # 交叉生成新子代种群
        childPop = gen_ops.crossover(chrom1, chrom2, config.CROSS_TIMES)
        # 子代种群变异
        childPop = gen_ops.mutate(childPop, config.P_MUT)
        # 将子代种群应用到 best_solution, 获取所有新解
        new_solutions, new_times = gen_ops.applyPopulation(childPop, best_solution, parameters)
        # 计算子代适应度
        fitness = gen_ops.fitness(new_times)
        # 接受更优接
        for i in range(len(new_solutions)):
            new_time = new_times[i]
            if new_time < best_time:
                best_time = new_time
                best_solution = new_solutions[i]
                print('new best time: ', best_time)
        #子代选择两个个体
        childChrom1 = gen_ops.bestSelection(childPop, fitness)
        childChrom2 = gen_ops.rouletteSelection(childPop, fitness)
        #替换所选的父代个体
        population[parentIdx1], population[parentIdx2] = childChrom1, childChrom2
    print('final best time: ', best_time)
    return best_time

def runForTest(problem):
    parameters = parse(problem)
    best_solution = initializeResult(parameters)
    best_time = timeTaken(best_solution, parameters)

    #初始化种群
    population = gen_ops.initPopulation(config.POP_SIZE, config.CHROM_LENGTH)
    #开始循环
    for i in range(config.GEN_NUM):
        # 随机选出两个父代
        parentIdx1, parentIdx2 = gen_ops.randomSelection(population)
        chrom1, chrom2 = population[parentIdx1], population[parentIdx2]
        # 交叉生成新子代种群
        childPop = gen_ops.crossover(chrom1, chrom2, config.CROSS_TIMES)
        # 子代种群变异
        childPop = gen_ops.mutate(childPop, config.P_MUT)
        # 将子代种群应用到 best_solution, 获取所有新解
        new_solutions, new_times = gen_ops.applyPopulation(childPop, best_solution, parameters)
        # 计算子代适应度
        fitness = gen_ops.fitness(new_times)
        # 接受更优接
        for i in range(len(new_solutions)):
            new_time = new_times[i]
            if new_time < best_time:
                best_time = new_time
                best_solution = new_solutions[i]
        #子代选择两个个体
        childChrom1 = gen_ops.bestSelection(childPop, fitness)
        childChrom2 = gen_ops.rouletteSelection(childPop, fitness)
        #替换所选的父代个体
        population[parentIdx1], population[parentIdx2] = childChrom1, childChrom2
    return best_time

if __name__ == '__main__':
    run(config.PROBLEM_PATH)
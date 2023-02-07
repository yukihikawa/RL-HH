import random

from src.LLH.LLHolder import LLHolder, LLHolder2
from src.LLH.LLHUtils import timeTaken


LLH = LLHolder2()

# 初始化高层启发式种群
def initPopulation(popSize, chromLength):
    """Initialize population."""
    population = []
    for i in range(popSize):
        population.append([random.randint(0, len(LLH) - 1) for j in range(chromLength)])
    return population

# 将染色体代表的 LHH 序列应用到 Solution 上,solution 是一个 (os, ms) 元组
# 不修改原解,返回新解
def applyChrom(chrom, solution, parameters):
    """Apply chromosome to solution."""
    (os, ms) = solution
    # 声明一个长度和 LLH 相同的禁忌列表
    #taboo = [0 for i in range(len(LLH))]
    prev_solution = best_solution = (os.copy(), ms.copy())
    prev_time = b_time = timeTaken(best_solution, parameters)
    for i in chrom:
        # if sum(taboo) == len(LLH):
        #     taboo = [0 for i in range(len(LLH))]
        # if taboo[i] == 1:
        #     continue
        cand_solution = LLH[i](prev_solution, parameters)
        cand_time = timeTaken(cand_solution, parameters)
        if cand_time < prev_time:
            prev_solution = cand_solution
            prev_time = cand_time
            #taboo = [0 for i in range(len(LLH))]
            if cand_time < b_time:
                best_solution = cand_solution
                b_time = cand_time
                #return best_solution, b_time
        else:
            #taboo[i] = 1
            p = random.random()
            if p > 0.5 +  0.01 * i:
                prev_solution = cand_solution
                prev_time = cand_time
    #print('bt: ', b_time)
    return best_solution, b_time

def applyChromTabu(chrom, solution, parameters):
    """Apply chromosome to solution."""
    (os, ms) = solution
    # 声明一个长度和 LLH 相同的禁忌列表
    taboo = [0 for i in range(len(LLH))]
    prev_solution = best_solution = (os.copy(), ms.copy())
    prev_time = b_time = timeTaken(best_solution, parameters)
    for i in chrom:
        if sum(taboo) == len(LLH):
            taboo = [0 for i in range(len(LLH))]
        if taboo[i] == 1:
            continue
        cand_solution = LLH[i](prev_solution, parameters)
        cand_time = timeTaken(cand_solution, parameters)
        if cand_time < prev_time:
            prev_solution = cand_solution
            prev_time = cand_time
            taboo = [0 for i in range(len(LLH))]
            if cand_time < b_time:
                best_solution = cand_solution
                b_time = cand_time
                #return best_solution, b_time
        else:
            taboo[i] = 1
            p = random.random()
            if p > 0.5 +  0.01 * i:
                prev_solution = cand_solution
                prev_time = cand_time
    #print('bt: ', b_time)
    return best_solution, b_time

#将一个种群的染色体应用到解上,获取包含所有新解的列表
def applyPopulation(population, solution, parameters):
    """Apply population to solution."""
    new_solutions = []
    new_times = []
    for i in range(len(population)):
        ns, nt = applyChrom(population[i], solution, parameters)
        #print('nt: ', nt, 'ns: ', ns)
        new_solutions.append(ns)
        new_times.append(nt)
    #print(new_times)
    return new_solutions, new_times

# 适应度,返回种群的适应度列表
def fitness(times):
    """Calculate fitness."""
    fitness = []
    for time in times:
        fitness.append(1 / time)
    return fitness



# 选择操作
# 从种群中随机选择两个个体
def randomSelection(population):
    """Select chromosome."""
    return random.sample(range(0, len(population)), 2)
# 根据适应度, 轮盘赌选择一个染色体
def rouletteSelection(population, fitness):
    """Select chromosome."""
    sum_fit = sum(fitness)
    # print(sum_fit)
    # print(fitness)
    # print(population)
    # print(random.random())
    for i, value in enumerate(fitness):
        # print(value)
        if random.random() < value / sum_fit:
            return population[i]
    return population[-1]

# 选择适应度最佳的个体
def bestSelection(population, fitness):
    """Select chromosome."""
    idx = fitness.index(max(fitness))
    selected = population[idx].copy()
    del population[idx]
    del fitness[idx]
    return selected


# 交叉操作
# 使用等位基因交叉, 返回两个新染色体
def crossoverTP(chrom1, chrom2):
    pos1 = random.randint(0, len(chrom1) - 1)
    pos2 = random.randint(0, len(chrom2) - 1)
    if pos1 > pos2:
        pos1, pos2 = pos2, pos1
    newChrom1 = chrom1[0:pos1] + chrom2[pos1:pos2] + chrom1[pos2:]
    newChrom2 = chrom2[0:pos1] + chrom1[pos1:pos2] + chrom2[pos2:]
    return newChrom1, newChrom2

# 用给定的两个染色体进行 2n 次操作, 返回新的候选种群
def crossover(chrom1, chrom2, cross_times):
    new_population = []
    for i in range(cross_times):
        newChrom1, newChrom2 = crossoverTP(chrom1, chrom2)
        new_population.append(newChrom1)
        new_population.append(newChrom2)
    return new_population

# 变异操作
#对传入的单个染色体进行变异
def singleMutate(chrom, pm):
    new_chrom = chrom.copy()
    for i in range(len(new_chrom)):
        if random.random() < pm:
            new_chrom[i] = random.randint(0, len(LLH) - 1)
    return new_chrom

#对传入的种群执行变异操作
def mutate(population, pm):
    new_population = []
    for i in range(len(population)):
        new_population.append(singleMutate(population[i], pm))
    return new_population
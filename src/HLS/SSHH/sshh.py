# 复现序列选择超启发式
import math
import random

import numpy as np
from src.LLH import LLHSet1 as llh
from src.LLH.LLHolder import LLHolder

# 定义一个类,维护一个有 10 个状态的状态转移矩阵,来表示从该状态转移到其他状态的概率,初始值全部为 1
class SequenceSelection:
    def __init__(self, llh_set):
        self.heuristics = llh_set
        self.transition_matrix = np.ones((len(self.heuristics), len(self.heuristics)))

        self.prevState = random.randint(0, len(self.heuristics) - 1)
        self.FLAG = 1
        self.prevTime = 10000
        self.bestTime = 10000
        self.best_solution = None
        #self.prev_solution = None

    # 定义一个函数,该函数接受一个状态,根据状态转移矩阵,返回一个状态
    # 从当前状态i转移到下一个状态j的概率定义为矩阵第 i 行第 j 列的值除以第 i 行所有元素之和
    # 使用轮盘赌算法选出nextState
    def next_state(self, current_state):
        transition_probabilities = self.transition_matrix[current_state, :] / np.sum(self.transition_matrix[current_state, :])
        next_state = np.random.choice(np.arange(self.transition_matrix.shape[0]), p=transition_probabilities)
        return next_state

    # 定义一个函数,接受 currentState 和 nextState,将状态转移矩阵中对应的值+1
    # nextState列的所有元素+1
    def update_transition_matrix(self, current_state, next_state):
        self.transition_matrix[current_state, next_state] += 1
        # nextState 列的所有元素+1
        self.transition_matrix[:, next_state] += 1


    # 定义一个函数, 接受solution, parameters, 根据prevState状态转移矩阵得出新 state;
    # 调用 heuristic[state] 对 solution 进行操作,返回新的 solution
    # 之后使用llh.timeTaken()比较两个 Solution, 使用改进的模拟退火接受,如果新解更优则直接接受,否则以一定概率接受
    # 接收概率由 FLAG 变量决定,当非更优解未被接受时,FLAG+1, FLAG 越大,非改进解被接受的概率越高.当解被接受时,FLAG=1
    # 仅当改进解被接受时更新状态转移矩阵
    # 更新 prevState, 返回新的 solution
    def update_solution(self, solution, parameters):
        nextState = self.next_state(self.prevState)
        #print('llh called: ', nextState)
        new_solution = self.heuristics[nextState](solution, parameters)
        #prevTime = llh.timeTaken(solution, parameters)
        newTime = llh.timeTaken(new_solution, parameters)

        self.prevState = nextState
        if newTime < self.prevTime:

            self.update_transition_matrix(self.prevState, nextState)
            self.prevTime = newTime
            #self.prev_solution = new_solution
            # self.prevState = nextState
            self.FLAG = 1
            if self.bestTime > newTime:
                print('newTime: ', newTime, 'prevTime: ', self.prevTime, 'beat time', self.bestTime)
                self.bestTime = newTime
                self.best_solution = new_solution

            return new_solution
        else:
            p = random.random()
            temp = np.exp(-(newTime - self.prevTime) / (self.FLAG * 0.01))
            #print('p: ', p, 'temp: ', temp)
            #if (p < temp) & (newTime > self.prevTime):
            if p < temp:
                #print('accepted!')
                self.prevTime = newTime
                self.FLAG = 1
                # self.prevState = nextState
                return new_solution

            else:
                self.FLAG += 1
                return solution



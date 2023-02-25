import math
import random

import gym
import numpy as np
from gym import spaces

import src.LLH.LLHUtils as llh
import src.utils.encoding as encoding
from src.HLS.DQN2_ERL_VNS.train import config
from src.LLH.LLHSetVNS import LLHSetVNS
from src.LLH.LLHolder import LLHolder
from src.utils import parser


#10个问题的最优解区间
IDEAL_TIME = {'MK01': (36, 40), 'MK02': (27, 30), 'MK03': (204, 2204), 'MK04': (60, 65), 'MK05': (168, 178), 'MK06': (60, 80), 'MK07': (133, 157), 'MK08': (523, 523), 'MK09': (299, 330), 'MK10': (165, 280)}
BEST_TIME = {'MK01': 40, 'MK02': 26, 'MK03': 203, 'MK04': 60, 'MK05': 171, 'MK06': 57, 'MK07': 139, 'MK08': 522, 'MK09': 307, 'MK10': 250}

class vns_env2(gym.Env):
    def __init__(self, problem = '', problem_path = '', llh_set = 1, solve_iter = 5000, train = True):
        self.train = train
        self.problem = problem
        self.problem_path = problem_path
        self.solve_iter = solve_iter
        self.vns = LLHSetVNS()
        self.heuristics = self.vns.llh
        # 定义动作空间, LLH方法数量
        self.action_space = spaces.Discrete(len(self.heuristics))

        # 定义状态空间
        low = np.array([20, 0, -float('inf')])
        high = np.array([40, self.solve_iter, float('inf')])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        action_c = action
        # 获取原本的时间,用于评估
        previous = self.vns.previous_time
        best = self.vns.best_time
        # 记录调用
        self.callCount[action_c] += 1
        self.ITER += 1
        # 执行
        self.heuristics[action_c]()
        # 执行完毕,获取新实践
        newPrevious = self.vns.previous_time
        newBest = self.vns.best_time
        # 更新未改进回合数
        if newPrevious < previous:
            self.NOT_IMPROVED = 1
        else:
            self.NOT_IMPROVED += 1

        if newBest < best:
            print('ITER:', self.ITER, 'best improved: ', best, '->', newBest)
            self.NOT_IMPROVED_BEST = 1
        else:
            self.NOT_IMPROVED_BEST += 1

        termination = self.termination()
        # 奖励
        reward = self.rewardA(previous, best, newPrevious, newBest, termination)

        #if action_c in [3, 5, 6]:
        if action_c in [0, 1, 2, 3, 4]:
            cla = 20
        else:
            cla = 40
        delta = previous - newPrevious #局部最优判定

        s_ = np.array([cla, self.NOT_IMPROVED, delta])

        if termination:
            self.render()
            return s_, reward, termination, {'bestTime': self.vns.best_time}
        return s_, reward, termination, {}

    # 奖励函数
    def reward(self, previous, best, new_previous, new_best, terminal = None):
        if previous > new_previous:
            reward = 10
            self.rewardImpL += reward
            if best > new_best:
                reward += 100
                self.rewardImpP += reward
        elif previous == new_previous:
            if self.NOT_IMPROVED < 40:
                reward = 0
            else:
                reward = -1
            self.rewardSta += reward
        else:
            if self.NOT_IMPROVED < 40:
                reward = -1
            else:
                reward = 10
            self.rewardMut += reward
        return reward

    def rewardA(self, previous, best, new_previous, new_best, terminal = None):
        if previous > new_previous:
            reward = 10
            self.rewardImpL += reward
            if best > new_best:
                reward += 100
                self.rewardImpP += reward
        elif previous == new_previous:
            reward = -1
            self.rewardSta += reward
        else:
            reward = 1
            self.rewardMut += reward
        if terminal:
            scale = 1000 #控制峰值
            base = 0.95  # 控制指数衰减的基数，可以调整以改变函数的形状
            offset = self.TARGET  # 控制函数在Target处的取值，可以调整以改变函数的峰值
            reward += (1 + math.exp(-base * (self.vns.best_time - offset) ** 2)) * scale
        return reward

    # 终止条件
    def termination(self):
        #if self.bestTime in range(IDEAL_TIME[self.problem][0], IDEAL_TIME[self.problem][1]) or self.ITER > 5000:
        if self.ITER > self.solve_iter:
            return True
        else:
            return False

    def reset(self, **kwargs):
        # print('llh_set: ', self.llh_set)
        # print("heuristics: ", self.heuristics)
        self.vns.train = self.train
        print('train: ', self.vns.train)
        self.TERMINATION_TIME = IDEAL_TIME[self.problem][1]
        self.TARGET = BEST_TIME[self.problem]
        self.vns.reset(self.problem_path) # 重设底层 LLH
        self.original_time = self.vns.previous_time
        # print(self.best_solution[0])
        # print(self.best_solution[1])
        self.NOT_IMPROVED_BEST = 1
        self.NOT_IMPROVED = 1
        self.rewardImpP = 0
        self.rewardImpL = 0
        self.rewardMut = 0
        self.rewardSta = 0
        self.rewardEnd = 0
        self.ITER = 1

        self.prevState = random.randint(0, len(self.heuristics))
        self.callCount = [0 for i in range(len(self.heuristics))]
        self.improveCount = [0 for i in range(len(self.heuristics))]
        # 返回一个一维的整型张量,随机取值,取值范围是[0,10)
        return np.array([self.prevState, self.NOT_IMPROVED, 0])

    def render(self, mode='human'):
        print('origin time', self.original_time)
        print('LLH called: ')
        print(self.callCount)
        print('improve called: ')
        print(self.improveCount)
        print('reward from pure improve: ', self.rewardImpP)
        print('reward from l improve: ', self.rewardImpL)
        print('reward from stay: ', self.rewardSta)
        print('reward from mutation: ', self.rewardMut)
        print('reward from end: ', self.rewardEnd)
        print('total reward', self.rewardImpP + self.rewardImpL + self.rewardSta + self.rewardMut + self.rewardEnd)
        print("finish time: ", self.vns.best_time)
        print('====================================================')
        # gantt_data = decoding.translate_decoded_to_gantt(decoding.decode(self.parameters, self.best_solution[0], self.best_solution[1]))
        # gantt.draw_chart(gantt_data)

    def close(self):
        pass

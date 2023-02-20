import math
import random

import gym
import numpy as np
from gym import spaces

import src.LLH.LLHUtils as llh
import src.utils.encoding as encoding
from src.HLS.DQN2_ERL_VNS.train import config
from src.LLH.LLHolder import LLHolder
from src.utils import parser


#10个问题的最优解区间
IDEAL_TIME = {'MK01': (36, 40), 'MK02': (27, 30), 'MK03': (204, 2204), 'MK04': (60, 65), 'MK05': (168, 178), 'MK06': (60, 80), 'MK07': (133, 157), 'MK08': (523, 523), 'MK09': (299, 330), 'MK10': (165, 280)}
BEST_TIME = {'MK01': 40, 'MK02': 26, 'MK03': 204, 'MK04': 60, 'MK05': 171, 'MK06': 57, 'MK07': 139, 'MK08': 523, 'MK09': 307, 'MK10': 197}

class hh_env(gym.Env):
    def __init__(self, problem = '', problem_path = '', llh_set = 1, solve_iter = 1000, train = True):
        # self.factory = parser.parse(PROBLEM_STR)
        # self.solution = encoding.initializeResult(self.factory)
        # self.iter = 0
        # self.prevTime = 0
        self.train = train
        self.problem = problem
        self.problem_path = problem_path
        self.llh_set = llh_set
        self.solve_iter = solve_iter
        self.holder = LLHolder(self.llh_set)
        self.heuristics = self.holder.set.llh
        # 定义动作空间,状态数等于heuristics的数量
        self.action_space = spaces.Discrete(len(self.heuristics))

        # 定义状态空间
        low = np.array([0.0, 0, -float('inf')])
        high = np.array([len(self.heuristics), self.solve_iter, float('inf')])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        self.callCount[action] += 1
        self.ITER += 1
        newSolution = self.heuristics[action](self.prevSolution, self.parameters)
        newTime = llh.timeTaken(newSolution, self.parameters)
        termination = self.termination()
        # 奖励
        reward = self.reward(newTime, termination)
        # 状态和解的接受
        delta = self.prevTime - newTime#局部最优判定
        self.acceptA(newTime, newSolution, action)
        s_ = np.array([action, self.NOT_IMPROVED, delta])
        info = {}
        if termination:
            self.render()
            info = {'bestTime': self.bestTime}
        return s_, reward, termination, info


    def accept(self, newTime, newSolution, action):
        if self.bestTime > newTime:
            self.improveCount[action] += 1
            self.NOT_IMPROVED = 1
            self.bestSolution = newSolution
            self.bestTime = newTime
        else:
            self.NOT_IMPROVED += 1

        if newTime <= self.prevTime:
            self.NOT_ACCEPTED = 1
            self.prevTime = newTime
            self.prevSolution = newSolution
        else:
            print('self.NOT_ACCEPTED: ', self.NOT_ACCEPTED)
            if random.random() < math.exp(-(newTime - self.prevTime) / (self.NOT_ACCEPTED *0.01)):
                self.prevTime = newTime
                self.prevSolution = newSolution
                self.NOT_ACCEPTED = 1
            else:
                self.NOT_ACCEPTED += 1

    def acceptA(self, newTime, newSolution, action):
        # print("train", self.train, "iter: ", self.ITER, "new bestTime: ", self.bestTime, 'llh called: ', action)
        if self.prevTime > newTime:
            self.improveCount[action] += 1
            self.solution = newSolution
            self.prevTime = newTime
            if (self.bestTime > newTime):
                self.best_solution = newSolution
                self.bestTime = newTime
                # print("train", self.train, "iter: ", self.ITER, "new bestTime: ", self.bestTime, 'llh called: ', action)
            self.NOT_ACCEPTED = 1
            self.NOT_IMPROVED = 1
        elif self.prevTime < newTime:
            self.NOT_IMPROVED += 1
            # print(' ')
            # 解的接受
            p = random.random()
            # 模拟退火
            temp = math.exp(-(newTime - self.prevTime) / (self.NOT_ACCEPTED * 0.01))
            #
            if p < temp:
                # print('accepted!!!!!!!!!!!!!!!!!!!!!!!!!!')
                # print('NOT_IMPROVED: ', self.NOT_IMPROVED, 'temp: ', temp, 'p: ', p)
                # print('ori prevTime: ', self.prevTime, 'newTime: ', newTime, 'llh called: ', action)
                self.solution = newSolution
                self.prevTime = newTime
                self.NOT_ACCEPTED = 1
                self.NOT_IMPROVED += 1
            else:
                # print('declined!!!!!!!!!')
                self.NOT_ACCEPTED += 1
                self.NOT_IMPROVED += 1
        else:
            self.solution = newSolution
            self.prevTime = newTime
            # print('new prevTime: ', self.prevTime, 'newTime: ', newTime)
            self.NOT_ACCEPTED += 1
            self.NOT_IMPROVED += 1

    def reward(self, newTime, termination):
        if newTime < self.bestTime: #主线任务奖励
            reward = 100 * (self.bestTime - newTime) * (self.oriTime / newTime)
            self.rewardImpP += reward
        elif newTime >= self.bestTime & newTime < self.prevTime:
            reward = -1
            self.rewardImpL += reward
        elif newTime == self.prevTime:
            reward = -5
            self.rewardSta += reward
        else:
            if random.random() < math.exp(-(newTime - self.prevTime) / (self.NOT_ACCEPTED *0.01)):
                reward = -1
            else:
                reward = -10
            self.rewardMut += reward
        if termination:
            if self.bestTime <= self.TERMINATION_TIME:
                reward += 2000
                self.rewardEnd += reward
        return reward

    def termination(self):
        #if self.bestTime in range(IDEAL_TIME[self.problem][0], IDEAL_TIME[self.problem][1]) or self.ITER > 5000:
        if self.ITER > self.solve_iter:
            return True
        elif self.bestTime <= self.TERMINATION_TIME:
            return True
        else:
            return False
    def reset(self, **kwargs):
        self.heuristics = LLHolder(self.llh_set).set.llh
        # print('llh_set: ', self.llh_set)
        # print("heuristics: ", self.heuristics)
        self.TERMINATION_TIME = IDEAL_TIME[self.problem][1]
        self.parameters = parser.parse(self.problem_path)
        self.bestSolution = self.prevSolution = encoding.initializeResult(self.parameters)
        self.oriTime = self.prevTime = self.bestTime = llh.timeTaken(self.prevSolution, self.parameters)
        # print(self.bestSolution[0])
        # print(self.bestSolution[1])
        self.NOT_ACCEPTED = 1
        self.NOT_IMPROVED = 1
        self.rewardImpP = 0
        self.rewardImpL = 0
        self.rewardMut = 0
        self.rewardSta = 0
        self.rewardEnd = 0
        self.ITER = 1

        # 循环 10 次
        # for i in range(10):
        #     print("original time: ", self.prevTime)
        self.prevState = random.randint(0, len(self.heuristics))
        self.callCount = [0 for i in range(len(self.heuristics))]
        self.improveCount = [0 for i in range(len(self.heuristics))]
        # 返回一个一维的整型张量,随机取值,取值范围是[0,10)
        return np.array([self.prevState, self.NOT_IMPROVED, 0])

    def render(self, mode='human'):
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
        print("finish time: ", self.bestTime)
        # gantt_data = decoding.translate_decoded_to_gantt(decoding.decode(self.parameters, self.best_solution[0], self.best_solution[1]))
        # gantt.draw_chart(gantt_data)

    def close(self):
        pass



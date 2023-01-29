import math
import random

import gym
import numpy as np
from gym import spaces

import src.LLH.LLHUtils as llh
import src.utils.encoding as encoding
from src.HLS.DQN_ERL import config
from src.LLH.LLHolder import LLHolder
from src.utils import parser

problem_str = config.PROBLEM


class hh_env(gym.Env):
    def __init__(self):
        # self.factory = parser.parse(PROBLEM_STR)
        # self.solution = encoding.initializeResult(self.factory)
        # self.iter = 0
        # self.prevTime = 0
        self.heuristics = LLHolder()
        # 定义动作空间
        self.action_space = spaces.Discrete(len(self.heuristics))
        # 定义状态空间, 有 0-9 共 10 个整数状态
        # self.observation_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32)

        # 定义状态空间
        # self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,))
        # self.observation_space = spaces.Tuple((spaces.Discrete(len(self.heuristics)), spaces.Discrete(5000), spaces.Box(-float('inf'), float('inf'), shape=(1,))))
        low = np.array([0.0, 0, -float('inf')])
        high = np.array([len(self.heuristics), 5000, float('inf')])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # self.observation_space = spaces.Discrete(10)
        # self.state = None
        # self.env_name = 'hh_env'  # the name of this env.
        self.prev_loss = 0

        # self.state_dim = self.observation_space.shape[0]  # feature number of state
        # self.action_dim = self.action_space.n  # feature number of action
        # self.if_discrete = True  # discrete action or continuous action

    def step(self, action):
        self.callCount[action] += 1
        self.ITER += 1
        termination = self.ITER > 5000
        if termination:
            self.render()
        # print("in env: action: ", action)
        # print(self.heuristics[action])
        newSolution = self.heuristics[action](self.solution, self.parameters)
        # prevTime = llh.timeTaken(self.solution, self.parameters)
        newTime = llh.timeTaken(newSolution, self.parameters)
        # 状态定义为

        # 奖励函数
        if self.prevTime > newTime:
            self.solution = newSolution
            self.prevTime = newTime
            if (self.bestTime > newTime):
                self.best_solution = newSolution
                self.bestTime = newTime
            reward = 3 + self.NOT_IMPROVED * 0.02
            self.rewardImp += reward
            self.NOT_ACCEPTED = 1
            self.NOT_IMPROVED = 1
        else:
            self.NOT_IMPROVED += 1
            if self.prevTime == newTime:
                reward = 0
            else:
                # reward = self.NOT_IMPROVED * 10 / self.ITER
                # reward = 2 * math.exp(-(35 / self.NOT_IMPROVED)) - 1
                reward = 0.1
                # if self.NOT_IMPROVED <= 8:
                #     reward = 0
                # else:
                #     reward = 0.005 * self.NOT_IMPROVED
                self.rewardMut += reward
                # reward = -1
                # print("mut reward: ", reward)

            # 解的接受
            p = random.random()
            temp = np.exp(-(newTime - self.prevTime) / (self.NOT_ACCEPTED * 0.01))
            if p < temp:
                # print('accepted!')
                self.solution = newSolution
                self.prevTime = newTime
                self.NOT_ACCEPTED = 1
                self.NOT_IMPROVED = 1
            else:
                self.NOT_ACCEPTED += 1
                self.NOT_IMPROVED += 1
        # print("finish time: ", self.prevTime)
        # 用 action 创建一个一维张量, 并且将其转换为整型
        # s_ = np.array([action], dtype=np.int32)
        if action in [3, 5, 6]:
            # if action in [0, 1, 2]:
            ck = 20
        else:
            ck = 40
        delta = (self.prevTime - newTime) / self.prevTime + ck #局部最优判定
        s_ = np.array([action, self.NOT_IMPROVED, delta])
        return s_, reward, termination, {}

    def reset(self, **kwargs):
        self.parameters = parser.parse(problem_str)
        self.best_solution = self.solution = encoding.initializeResult(self.parameters)
        self.NOT_ACCEPTED = 1
        self.NOT_IMPROVED = 1
        self.rewardImp = 0
        self.rewardMut = 0
        self.ITER = 1
        self.prevTime = self.bestTime = llh.timeTaken(self.solution, self.parameters)
        self.prevState = random.randint(0, 10)
        self.callCount = [0 for i in range(len(self.heuristics))]
        # 返回一个一维的整型张量,随机取值,取值范围是[0,10)
        return np.array([self.prevState, self.NOT_IMPROVED, 0])

    def render(self, mode='human'):
        print('LLH called: ')
        print(self.callCount)
        print('reward from improve: ', self.rewardImp)
        print('reward from mutation: ', self.rewardMut)
        print("finish time: ", self.bestTime)
        # gantt_data = decoding.translate_decoded_to_gantt(decoding.decode(self.parameters, self.best_solution[0], self.best_solution[1]))
        # gantt.draw_chart(gantt_data)

    def close(self):
        pass

    def stepTest(self, action):
        newSolution = self.heuristics[action](self.solution, self.parameters)
        # prevTime = llh.timeTaken(self.solution, self.parameters)
        newTime = llh.timeTaken(newSolution, self.parameters)
        # 奖励函数
        if self.prevTime > newTime:
            self.solution = newSolution
            self.prevTime = newTime
            if (self.bestTime > newTime):
                self.best_solution = newSolution
                self.bestTime = newTime

            self.NOT_ACCEPTED = 1
        else:
            # 解的接受
            p = random.random()
            temp = np.exp(-(newTime - self.prevTime) / (self.NOT_ACCEPTED * 0.01))
            if p < temp:
                print('accepted!')
                self.solution = newSolution
                self.prevTime = newTime
                self.NOT_ACCEPTED = 1
            else:
                self.NOT_ACCEPTED += 1
                print("NOT ACCEPTED count: ", self.NOT_ACCEPTED)

        if action in [3, 5, 6]:
            ck = 20
        else:
            ck = 40
        s_ = (self.prevTime - newTime) / self.prevTime + ck
        s_ = np.array([s_], dtype=np.float32)
        return s_, 0, self.ITER > 3000, {}

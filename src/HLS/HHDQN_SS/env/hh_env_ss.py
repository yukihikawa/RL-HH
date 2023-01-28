import math
import random

import gym
import numpy as np
from gym import spaces
import src.LLH.encoding as encoding
import src.LLH.decoding as decoding
from src.utils import parser, gantt
import src.LLH.lowlevelheuristic as llh
from src.HLH.HHDQN_SS import config

problem_str = config.PROBLEM

# "C:\\Users\emg\PycharmProjects\GenFJSP\src\HLH\HHDQN\env\Mk01.fjs"
class hh_env_ss(gym.Env):
    def __init__(self):
        #self.factory = parser.parse(PROBLEM_STR)
        #self.solution = encoding.initializeResult(self.factory)
        #self.iter = 0
        #self.prevTime = 0
        # 定义动作空间
        self.action_space = spaces.Discrete(10)
        # 定义状态空间, 有 0-9 共 10 个整数状态
        self.observation_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32)
        # self.observation_space = spaces.Discrete(10)
        # self.state = None
        self.env_name = 'hh_env_ss-v0'  # the name of this env.
        self.prev_loss = 0
        self.heuristics = llh.LLHolder()

        # self.state_dim = self.observation_space.shape[0]  # feature number of state
        # self.action_dim = self.action_space.n  # feature number of action
        # self.if_discrete = True  # discrete action or continuous action

    def step(self, action):
        #print("in env: action: ", action)
        #print(self.heuristics[action])
        newSolution = self.heuristics[action](self.solution, self.parameters)
        # prevTime = llh.timeTaken(self.solution, self.parameters)
        newTime = llh.timeTaken(newSolution, self.parameters)
        # 状态定义为


        # 奖励函数
        if self.prevTime > newTime:
            self.best_solution = self.solution = newSolution
            self.prevTime = newTime
            reward = 1
            self.FLAG = 1
        else:
            if self.prevTime == newTime:
                reward = 0
            else:
                reward = 0.2
            p = random.random()
            temp = np.exp(-(newTime - self.prevTime) / (self.FLAG * 0.01))
            if p < temp:
                #print('accepted!')
                self.solution = newSolution
                self.prevTime = newTime
                self.FLAG = 1
            else:
                self.FLAG += 1
        #print("finish time: ", self.prevTime)
        # 用 action 创建一个一维张量, 并且将其转换为整型
        s_ = np.array([action], dtype=np.int32)
        return s_, reward, False, {}

    def reset(self, **kwargs):
        self.parameters = parser.parse(problem_str)
        self.best_solution = self.solution = encoding.initializeResult(self.parameters)
        self.FLAG = 1
        self.prevTime = llh.timeTaken(self.solution, self.parameters)
        self.prevState = random.randint(0, 10)
        # 返回一个一维的整型张量,随机取值,取值范围是[0,10)
        return np.array([self.prevState], dtype=int)

    def render(self, mode='human'):
        time = llh.timeTaken(self.best_solution, self.parameters)
        print("finish time: ", time)
        #gantt_data = decoding.translate_decoded_to_gantt(decoding.decode(self.parameters, self.best_solution[0], self.best_solution[1]))
        #gantt.draw_chart(gantt_data)

    def close(self):
        pass
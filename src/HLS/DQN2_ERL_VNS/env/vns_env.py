import math
import random

import gym
import numpy as np
from gym import spaces

import src.LLH.LLHUtils as llh
import src.utils.encoding as encoding
from src.HLS.DQN2_ERL_VNS.train import config
from src.HLS.ILS.actionILS import action
from src.LLH.LLHSetILS import LLHSetILS
from src.LLH.LLHSetVNS import LLHSetVNS
from src.LLH.LLHolder import LLHolder
from src.utils import parser


#10个问题的最优解区间
IDEAL_TIME = {'MK01': (36, 40), 'MK02': (27, 30), 'MK03': (204, 2204), 'MK04': (60, 65), 'MK05': (168, 178), 'MK06': (60, 80), 'MK07': (133, 157), 'MK08': (523, 523), 'MK09': (299, 330), 'MK10': (165, 280)}
BEST_TIME = {'MK01': 40, 'MK02': 26, 'MK03': 203, 'MK04': 60, 'MK05': 171, 'MK06': 57, 'MK07': 139, 'MK08': 522, 'MK09': 307, 'MK10': 250}

class vns_env(gym.Env):
    def __init__(self, problem = '', problem_path = '', llh_set = 1, solve_iter = 5000, train = True):
        self.train = train
        self.time_limit = 0
        self.NoE = 0
        self.problem = problem
        self.problem_path = problem_path
        self.solve_iter = solve_iter
        self.action_manager = action()
        self.actions = self.action_manager.actions
        # 定义动作空间, LLH方法数量
        self.action_space = spaces.Discrete(len(self.actions))

        # 定义状态空间,维度为 1,取值范围 0-1
        # self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        low = np.array([0.0, ])
        high = np.array([1.0, ])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        action_c = action
        print("stage: ", self.STAGE, 'global best: ', self.action_manager.llh_manager.best_time, 'action: ', action_c, 'previous: ', self.action_manager.llh_manager.previous_time)

        # 获取原本的时间,用于评估
        previous = self.action_manager.llh_manager.previous_time
        prev_global_best = self.action_manager.llh_manager.best_time
        # 设计并执行
        self.action_manager.execute(action_c)
        print('global best: ', self.action_manager.llh_manager.best_time, 'new_previous: ', self.action_manager.llh_manager.previous_time)
        #终止条件
        termination = self.termination()
        # 奖励
        reward = self.reward(previous)
        # 更新既往 avg(fit)
        self.total_fitness += self.action_manager.llh_manager.previous_time
        self.STAGE += 1
        # 状态norm( f ) = f /Avg( fnew)
        s_ = self.action_manager.llh_manager.previous_time / (self.total_fitness / self.STAGE)

        if termination:
            self.render()
            return s_, reward, termination, {'bestTime': self.action_manager.llh_manager.best_time}
        return s_, reward, termination, {}

    # 奖励函数
    def reward(self, prev_global_best):
        if prev_global_best > self.action_manager.llh_manager.best_time:
            reward = 1
        else:
            reward = -1
        self.total_reward += reward
        return reward



    # 终止条件
    def termination(self):
        #if self.bestTime in range(IDEAL_TIME[self.problem][0], IDEAL_TIME[self.problem][1]) or self.ITER > 5000:
        if self.action_manager.check_exceed_time_limit():
            return True
        else:
            return False

    def reset(self, **kwargs):
        # print('llh_set: ', self.llh_set)
        # print("heuristics: ", self.heuristics)
        # self.vns.train = self.train
        # print('train: ', self.vns.train)
        self.TERMINATION_TIME = IDEAL_TIME[self.problem][1]
        self.TARGET = BEST_TIME[self.problem]
        self.action_manager.llh_manager.reset(self.problem_path) # 重设底层 LLH
        self.action_manager.set_time_start()
        self.action_manager.time_limit = self.time_limit
        self.action_manager.NoE = self.NoE
        # 初始fitness
        self.original_time = self.action_manager.llh_manager.previous_time
        # print(self.best_solution[0])
        # print(self.best_solution[1])
        # self.NOT_IMPROVED_BEST = 1
        # self.NOT_IMPROVED = 1
        # self.rewardImpP = 0
        # self.rewardImpL = 0
        # self.rewardMut = 0
        # self.rewardSta = 0
        # self.rewardEnd = 0
        self.total_reward = 0
        self.total_fitness = self.original_time
        self.STAGE = 1

        # self.prevState = random.randint(0, len(self.heuristics))
        # self.callCount = [0 for i in range(len(self.heuristics))]
        # self.improveCount = [0 for i in range(len(self.heuristics))]
        return np.array([1.0, ])

    def render(self, mode='human'):
        print('origin time', self.original_time)
        # print('LLH called: ')
        # print(self.callCount)
        # print('improve called: ')
        # print(self.improveCount)
        # print('reward from pure improve: ', self.rewardImpP)
        # print('reward from l improve: ', self.rewardImpL)
        # print('reward from stay: ', self.rewardSta)
        # print('reward from mutation: ', self.rewardMut)
        # print('reward from end: ', self.rewardEnd)
        # print('total reward', self.rewardImpP + self.rewardImpL + self.rewardSta + self.rewardMut + self.rewardEnd)
        print('total_reward', self.total_reward)
        print("finish time: ", self.action_manager.llh_manager.best_time)
        print('====================================================')
        # gantt_data = decoding.translate_decoded_to_gantt(decoding.decode(self.parameters, self.best_solution[0], self.best_solution[1]))
        # gantt.draw_chart(gantt_data)

    def close(self):
        pass

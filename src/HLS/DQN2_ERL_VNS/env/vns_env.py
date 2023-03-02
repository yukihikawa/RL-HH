import math
import random

import gym
import numpy as np
from gym import spaces
# import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from torch import nn

import src.LLH.LLHUtils as llh
import src.utils.encoding as encoding
from src.HLS.DQN2_ERL_VNS.train import config
from src.LLH.LLHSetVNS import LLHSetVNS
from src.LLH.LLHSetVNS2 import LLHSetVNS2
from src.LLH.LLHolder import LLHolder
from src.utils import parser


#10个问题的最优解区间
IDEAL_TIME = {'MK01': (36, 40), 'MK02': (27, 30), 'MK03': (204, 2204), 'MK04': (60, 65), 'MK05': (168, 178), 'MK06': (60, 80), 'MK07': (133, 157), 'MK08': (523, 523), 'MK09': (299, 330), 'MK10': (165, 280)}
BEST_TIME = {'MK01': 40, 'MK02': 26, 'MK03': 203, 'MK04': 60, 'MK05': 171, 'MK06': 57, 'MK07': 139, 'MK08': 522, 'MK09': 307, 'MK10': 250}

class vns_env(gym.Env):
    def __init__(self, problem = '', problem_path = '', llh_set = 1, solve_iter = 5000, train = True):
        self.train = train
        self.problem = problem
        self.problem_path = problem_path
        self.solve_iter = solve_iter
        self.vns = LLHSetVNS2()
        self.heuristics = self.vns.llh
        # 定义动作空间, LLH方法数量
        self.action_space = spaces.Discrete(len(self.heuristics))

        # 定义状态空间
        low = np.array([20, 0, -float('inf')])
        high = np.array([40, self.solve_iter, float('inf')])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def roulette_selection(Self, data):
        # 计算概率分布
        total = sum(data)
        if total == 0:
            return random.randint(0, len(data) - 1)

        probabilities = [d / total for d in data]
        # 计算累计概率
        cum_probabilities = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]
        # print('cum_probabilities: ', cum_probabilities)
        selected_index = None
        rand = random.uniform(0, 1)
        for i, cum_prob in enumerate(cum_probabilities):
            if rand <= cum_prob:
                selected_index = i

                break
        return selected_index

    def step(self, action):
        # print('action in env:', action)
        # action_c = action
        # action_c = action.argmax(dim=1, keepdim=True)
        # print('action_c in env:', action_c)
        # 屏蔽无效动作
        # action_c = self.masked_action75(action)

        action_c = self.roulette_selection(action[0].detach().cpu().numpy())
        # print('action_c in env:', action_c)
        # 获取原本的时间,用于评估
        previous = self.vns.previous_time
        best = self.vns.best_time
        # 记录调用
        self.callCount[action_c] += 1
        self.ITER += 1
        # 执行
        self.heuristics[action_c]()
        # 执行完毕,获取新时间
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
        reward = self.reward(previous, best, newPrevious, newBest, termination)

        #if action_c in [3, 5, 6]:
        if action_c in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            cla = 20
            self.record_y.append(20 - action_c)
        else:
            cla = 40
            self.record_y.append(action_c - 10)
        self.record_x.append(self.ITER)
        delta = previous - newPrevious #局部最优判定

        # self.update_invalid_action75(action_c, delta)

        s_ = np.array([cla, self.NOT_IMPROVED, delta])

        if termination:
            self.render()
            return s_, reward, termination, {'bestTime': self.vns.best_time}
        return s_, reward, termination, {}

    # 奖励函数
    # 修改了负向扰动的 reward,与未改进回合数相关联
    def reward(self, previous, best, new_previous, new_best, terminal = None):
        T_LS = 80
        if previous > new_previous:
            reward = 10
            self.rewardImpL += reward
            if best > new_best:
                reward += 1000
                self.rewardImpP += reward
        elif previous == new_previous:
            if self.NOT_IMPROVED < T_LS:
                reward = 0
            else:
                reward = -(self.NOT_IMPROVED - T_LS) * 0.1
            self.rewardSta += reward
        else:
            # reward = 0.1 * self.NOT_IMPROVED
            if self.NOT_IMPROVED < T_LS:
                reward = -5
            else:
                reward = 10
                self.NOT_IMPROVED = 1
            self.rewardMut += reward

        # if terminal:
        #     scale = 1000 #控制峰值
        #     base = 0.95  # 控制指数衰减的基数，可以调整以改变函数的形状
        #     offset = self.TARGET  # 控制函数在Target处的取值，可以调整以改变函数的峰值
        #     self.rewardEnd = (1 + 10000000 * math.exp(-base * (self.vns.best_time - offset) ** 2)) * scale
        #     # print("reward base", (1 + math.exp(-base * (self.vns.best_time - offset) ** 2)))
        #     reward += self.rewardEnd
        return reward

    def rewarA(self, previous, best, new_previous, new_best, terminal = None):
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
            # reward = 0.1 * self.NOT_IMPROVED
            reward = 1
            self.rewardMut += reward

        # if terminal:
        #     scale = 1000 #控制峰值
        #     base = 0.95  # 控制指数衰减的基数，可以调整以改变函数的形状
        #     offset = self.TARGET  # 控制函数在Target处的取值，可以调整以改变函数的峰值
        #     self.rewardEnd = (1 + 10000000 * math.exp(-base * (self.vns.best_time - offset) ** 2)) * scale
        #     # print("reward base", (1 + math.exp(-base * (self.vns.best_time - offset) ** 2)))
        #     reward += self.rewardEnd
        return reward

    # 终止条件
    def termination(self):
        #if self.bestTime in range(IDEAL_TIME[self.problem][0], IDEAL_TIME[self.problem][1]) or self.ITER > 5000:
        if self.ITER > self.solve_iter:
            return True
        else:
            return False

    def reset(self, **kwargs):
        # 总回合计数
        self.ITER = 1
        # 训练模式,区分初始解生成策略
        self.vns.train = self.train
        print('train: ', self.vns.train)
        # 时间目标,用于确定结束奖励
        self.TERMINATION_TIME = IDEAL_TIME[self.problem][1]
        self.TARGET = BEST_TIME[self.problem]
        print('TARGET: ', self.TARGET)
        # 重设底层 LLH
        self.vns.reset(self.problem_path)
        self.original_time = self.vns.previous_time
        # 局部最优回合计数
        self.NOT_IMPROVED_BEST = 1
        self.NOT_IMPROVED = 1
        # 奖励分类计数
        self.rewardImpP = 0
        self.rewardImpL = 0
        self.rewardMut = 0
        self.rewardSta = 0
        self.rewardEnd = 0
        # 无效动作掩蔽记录
        self.tolerance = [0 for i in range(len(self.heuristics))]
        self.mask = [False for i in range(len(self.heuristics))]
        # 记录策略
        self.record_x = []
        self.record_y = []
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
        plt.scatter(self.record_x, self.record_y)
        plt.xlabel('ITER')
        plt.ylabel('strategy')
        plt.show()
        # gantt_data = decoding.translate_decoded_to_gantt(decoding.decode(self.parameters, self.best_solution[0], self.best_solution[1]))
        # gantt.draw_chart(gantt_data)

    def close(self):
        pass

    # 更新无效操作
    def update_invalid_action(self, action_c, delta):
        T_MAX = 3
        # local search阶段
        if action_c in [0, 1, 2, 3, 4, 5]:
            if (delta == 0):
                # 容忍回合数+1
                self.tolerance[action_c] += 1
                # 超过容忍回合, mask 该动作
                if self.tolerance[action_c] > T_MAX:
                    self.mask[action_c] = True
                    self.tolerance[action_c] = 0

            else:
                self.tolerance[action_c] = 0
                # self.mask[:5] = [False for i in range(5)]
            # 如果 mask 前五位全部为 True
            if all(self.mask[:6]):
                # 前五位重设为 False
                self.mask[:6] = [False for i in range(6)]
        else:
            # print('action_c: ', action_c, 'mask: ', self.mask)
            self.mask[action_c] = True
            # 如果后四位全部为 TRUE
            if all(self.mask[6:]):
                # 后四位重设为 False
                self.mask[6:] = [False for i in range(5)]

    # 屏蔽无效操作
    def masked_action(self, action):
        M = 0.000000001
        # 将 action 转换为列表
        action_c = action.argmax(dim=1, keepdim=True)
        masked_action = action[0].detach().cpu().numpy()
        # print(masked_action)
        # print("==============masked: ", self.mask)
        for i in range(len(self.mask)):
            if self.mask[i]:
                masked_action[i] = M
        # print('origin action', action_c)
        # print('mask:', self.mask)
        # print('action Med: ', masked_action)

        # 对numpy数组 masked_action 进行 softmax 归一化
        masked_action = np.exp(masked_action) / np.sum(np.exp(masked_action), axis=0)

        # print('action Med: ', masked_action)
        if action_c in [0, 1, 2, 3, 4, 5]:
            # 截取masked_action前五位作为新的masked_action
            m_action = masked_action[:6]
            # print('piece:', m_action)
            # 对numpy数组 masked_action 取最大值的索引
            m_action = np.argmax(m_action)
        else:
            # 截取masked_action后四位作为新的masked_action
            m_action = masked_action[6:]
            # print('piece:', m_action)
            # 使用 softmax方法归一化
            m_action = np.argmax(m_action) + 6
        # print('masked action: ', m_action)
        return m_action

    def update_invalid_action75(self, action_c, delta):
        T_MAX = 3
        # local search阶段
        if action_c in [0, 1, 2, 3, 4, 5, 6]:
            if (delta == 0):
                # 容忍回合数+1
                self.tolerance[action_c] += 1
                # 超过容忍回合, mask 该动作
                if self.tolerance[action_c] > T_MAX:
                    self.mask[action_c] = True
                    self.tolerance[action_c] = 0

            else:
                self.tolerance[action_c] = 0
                # self.mask[:5] = [False for i in range(5)]
            # 如果 mask 前五位全部为 True
            if all(self.mask[:7]):
                # 前五位重设为 False
                self.mask[:7] = [False for i in range(7)]
        else:
            # print('action_c: ', action_c, 'mask: ', self.mask)
            self.mask[action_c] = True
            # 如果后四位全部为 TRUE
            if all(self.mask[7:]):
                # 后四位重设为 False
                self.mask[7:] = [False for i in range(5)]

    # 屏蔽无效操作
    def masked_action75(self, action):
        M = 0.000000001
        # 将 action 转换为列表
        action_c = action.argmax(dim=1, keepdim=True)
        masked_action = action[0].detach().cpu().numpy()
        # print(masked_action)
        # print("==============masked: ", self.mask)
        for i in range(len(self.mask)):
            if self.mask[i]:
                masked_action[i] = M
        # print('origin action', action_c)
        # print('mask:', self.mask)
        # print('action Med: ', masked_action)

        # 对numpy数组 masked_action 进行 softmax 归一化
        masked_action = np.exp(masked_action) / np.sum(np.exp(masked_action), axis=0)

        # print('action Med: ', masked_action)
        if action_c in [0, 1, 2, 3, 4, 5, 6, 7]:
            # 截取masked_action前五位作为新的masked_action
            m_action = masked_action[:7]
            # print('piece:', m_action)
            # 对numpy数组 masked_action 取最大值的索引
            m_action = np.argmax(m_action)
        else:
            # 截取masked_action后四位作为新的masked_action
            m_action = masked_action[7:]
            # print('piece:', m_action)
            # 使用 softmax方法归一化
            m_action = np.argmax(m_action) + 7
        # print('masked action: ', m_action)
        return m_action
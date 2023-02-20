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

class vns_env(gym.Env):
    def __init__(self, problem = '', problem_path = '', llh_set = 1, solve_iter = 5000, train = True):
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
        #print('heuristics: ', self.heuristics)
        # 定义动作空间
        self.action_space = spaces.Discrete(len(self.heuristics))
        # 定义状态空间, 有 0-9 共 10 个整数状态
        # self.observation_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32)

        # 定义状态空间
        # self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,))
        # self.observation_space = spaces.Tuple((spaces.Discrete(len(self.heuristics)), spaces.Discrete(5000), spaces.Box(-float('inf'), float('inf'), shape=(1,))))
        low = np.array([0.0, 0, -float('inf')])
        high = np.array([len(self.heuristics), self.solve_iter, float('inf')])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # self.observation_space = spaces.Discrete(10)
        # self.state = None
        # self.env_name = 'hh_env'  # the name of this env.
        self.prev_loss = 0

        # self.state_dim = self.observation_space.shape[0]  # feature number of state
        # self.action_dim = self.action_space.n  # feature number of action
        # self.if_discrete = True  # discrete action or continuous action

    def step(self, action):
        #print('llh_set: ', self.llh_set)
        #print(len(self.heuristics))
        #action_c = np.argmax(action)
        action_c = action
        #print(action_c)
        #print(self.callCount)
        self.callCount[action_c] += 1
        self.ITER += 1
        newSolution = self.heuristics[action_c](self.solution, self.parameters)
        # prevTime = llh.timeTaken(self.solution, self.parameters)
        newTime = llh.timeTaken(newSolution, self.parameters)
        #termination = self.ITER > 5000 or newTime == self.TERMINATION_TIME
        termination = self.terminationA()

        # 奖励
        reward = self.rewardNeoA(newTime, termination)
        #reward = self.reward_function(newTime)
        #print(" newTime: ", newTime)
        # 解的接受
        self.accept(newTime, newSolution, action_c)


        #if action_c in [3, 5, 6]:
        if action_c in [0, 1, 2]:
            ck = 20
        else:
            ck = 40
        delta = (self.prevTime - newTime) / self.prevTime + ck #局部最优判定
        s_ = np.array([action_c, self.NOT_IMPROVED, delta])

        if termination:
            #reward = self.endReward4A()
            self.render()
            return s_, reward, termination, {'bestTime': self.bestTime}
        return s_, reward, termination, {}

    def accept(self, newTime, newSolution, action):
        #print("train", self.train, "iter: ", self.ITER, "new bestTime: ", self.bestTime, 'llh called: ', action)
        if self.prevTime > newTime:
            self.improveCount[action] += 1
            self.solution = newSolution
            self.prevTime = newTime
            if (self.bestTime > newTime):
                self.best_solution = newSolution
                self.bestTime = newTime
                #print("train", self.train, "iter: ", self.ITER, "new bestTime: ", self.bestTime, 'llh called: ', action)
            self.NOT_ACCEPTED = 1
            self.NOT_IMPROVED = 1
        elif self.prevTime < newTime:
            self.NOT_IMPROVED += 1
            #print(' ')

            # 非改进解的接受
            p = random.random()
            #模拟退火
            temp = math.exp(-(newTime - self.prevTime) / (self.NOT_ACCEPTED *0.01))
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
                #print('declined!!!!!!!!!')
                self.NOT_ACCEPTED += 1
                self.NOT_IMPROVED += 1
        else:
            self.solution = newSolution
            self.prevTime = newTime
            #print('new prevTime: ', self.prevTime, 'newTime: ', newTime)
            self.NOT_ACCEPTED += 1
            self.NOT_IMPROVED += 1

    def acceptA(self, newTime, newSolution, action):
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

    def reward3A(self, newTime, terminal = None):


        if self.prevTime > newTime:
            if self.bestTime > newTime:
                #reward =  3 + 1 / newTime
                reward = (self.bestTime - newTime) + 2 / newTime
                reward *= 1.3
                #reward =  (self.bestTime - newTime) * 1.5 + 2 / newTime
                self.rewardImpP += reward
            else:
                reward = 1 / newTime
                self.rewardImpL += reward
            # print("imp ", self.rewardImp)
        else:
            if self.prevTime == newTime:
                reward = -0.03 + 1 / self.bestTime
                self.rewardSta += reward
            else:
                # reward = self.NOT_IMPROVED * 10 / self.ITER
                # reward = 2 * math.exp(-(35 / self.NOT_IMPROVED)) - 1
                reward = -0.02 + 1 / self.bestTime
                # if self.NOT_IMPROVED <= 8:
                #     reward = 0
                # else:
                #     reward = 0.005 * self.NOT_IMPROVED
                self.rewardMut += reward
                # reward = -1
                # print("mut reward: ", reward)
        return reward

    def rewardNeoA(self, newTime, termination):
        if self.prevTime > newTime:
            if self.bestTime > newTime:
                reward = (self.bestTime - newTime) + 3 / newTime
                reward *= 300
                self.rewardImpP += reward
            else:
                reward = 1 / newTime
                reward *= 5
                self.rewardImpL += reward
        else:
            if self.prevTime == newTime:
                reward = -0.03 + 1 / self.bestTime
                reward *= 10
                self.rewardSta += reward
            else:
                reward = -0.02 + 1 / self.bestTime
                reward *= 5
                self.rewardMut += reward
        if termination:
            if self.bestTime <= self.TERMINATION_TIME:
                reward += 1000
                self.rewardEnd += reward
        return reward

    def rewardNeo(self, newTime, termination):
        if newTime < self.bestTime: #主线任务奖励
            reward = 10 * (self.bestTime - newTime) * (self.oriTime / newTime)
            # reward = 100
            self.rewardImpP += reward
        elif newTime >= self.bestTime & newTime < self.prevTime:
            reward = 0.5
            self.rewardImpL += reward
        elif newTime == self.prevTime:
            reward = 0
            self.rewardSta += reward
        else:
            if random.random() < math.exp(-(newTime - self.prevTime) / (self.NOT_ACCEPTED *0.01)):
                reward = 10
            else:
                reward = 2
            self.rewardMut += reward
        if termination:
            if self.bestTime <= self.TERMINATION_TIME:
                reward += 2000
                self.rewardEnd += reward
        return reward

    def terminationA(self):
        #if self.bestTime in range(IDEAL_TIME[self.problem][0], IDEAL_TIME[self.problem][1]) or self.ITER > 5000:
        if self.ITER > self.solve_iter:
            return True
        else:
            return False
    def reset(self, **kwargs):
        self.heuristics = LLHolder(self.llh_set).set.llh
        # print('llh_set: ', self.llh_set)
        # print("heuristics: ", self.heuristics)
        self.TERMINATION_TIME = IDEAL_TIME[self.problem][1]
        self.parameters = parser.parse(self.problem_path)
        self.best_solution = self.solution = encoding.initializeResult(self.parameters)
        # print(self.best_solution[0])
        # print(self.best_solution[1])
        self.NOT_ACCEPTED = 1
        self.NOT_IMPROVED = 1
        self.rewardImpP = 0
        self.rewardImpL = 0
        self.rewardMut = 0
        self.rewardSta = 0
        self.rewardEnd = 0
        self.ITER = 1
        self.oriTime = self.prevTime = self.bestTime = llh.timeTaken(self.solution, self.parameters)
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

    def test(self):
        print('problem', self.problem)

    def reward3(self, newTime):
        if self.prevTime > newTime:
            reward =  self.TERMINATION_TIME / (newTime - self.TERMINATION_TIME + 1)
            self.rewardImp += reward
            # print("imp ", self.rewardImp)
        else:
            if self.prevTime == newTime:
                reward = -0.03
                self.rewardSta += reward
            else:
                # reward = self.NOT_IMPROVED * 10 / self.ITER
                # reward = 2 * math.exp(-(35 / self.NOT_IMPROVED)) - 1
                reward = -0.01
                # if self.NOT_IMPROVED <= 8:
                #     reward = 0
                # else:
                #     reward = 0.005 * self.NOT_IMPROVED
                self.rewardMut += reward
                # reward = -1
                # print("mut reward: ", reward)
        return reward

    def endReward4(self):
        #BestTime越小,奖励值越高
        if (self.bestTime in range(IDEAL_TIME[self.problem][0], IDEAL_TIME[self.problem][1] + 1)):
            finishRewardRate = 150 - (self.bestTime - IDEAL_TIME[self.problem][0]) / (IDEAL_TIME[self.problem][1] - IDEAL_TIME[self.problem][0] + 1) * 50
        else:
            finishRewardRate = 100 - (self.bestTime - IDEAL_TIME[self.problem][1]) / (IDEAL_TIME[self.problem][1] - IDEAL_TIME[self.problem][0] + 1) * 200
        #print("end reward: ", finishRewardRate)
        self.rewardEnd = finishRewardRate
        return finishRewardRate

    def endReward4A(self):
        #BestTime越小,奖励值越高
        base = int(math.ceil(self.bestTime / 100.0)) * 100
        finishRewardRate = 200 * (base / self.bestTime)
        self.rewardEnd = finishRewardRate
        #print("end reward: ", finishRewardRate)
        return finishRewardRate

    def termination(self):
        #if self.bestTime in range(IDEAL_TIME[self.problem][0], IDEAL_TIME[self.problem][1]) or self.ITER > 5000:
        if self.bestTime <= IDEAL_TIME[self.problem][0] or self.ITER > self.solve_iter:
            return True
        else:
            return False
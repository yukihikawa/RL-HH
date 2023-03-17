import math
import random

import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt

import src.LLH.LLHUtils as llh
import src.utils.encoding as encoding
from src.HLS.DQN2_ERL.train import config
from src.LLH.LLHolder import LLHolder
from src.utils import parser


#10个问题的最优解区间
IDEAL_TIME = {'MK01': (36, 42), 'MK02': (27, 32), 'MK03': (204, 211), 'MK04': (60, 81), 'MK05': (168, 186), 'MK06': (60, 86), 'MK07': (133, 157), 'MK08': (523, 523), 'MK09': (299, 369), 'MK10': (165, 296)}
BEST_TIME = {'MK01': 40, 'MK02': 26, 'MK03': 204, 'MK04': 60, 'MK05': 171, 'MK06': 57, 'MK07': 139, 'MK08': 523, 'MK09': 307, 'MK10': 197}

class hh_env(gym.Env):
    def __init__(self, problem = '', problem_path = '', llh_set = 1, solve_iter = 5000):
        # self.factory = parser.parse(PROBLEM_STR)
        # self.solution = encoding.initializeResult(self.factory)
        # self.iter = 0
        # self.prevTime = 0
        self.problem = problem
        self.problem_path = problem_path
        self.llh_set = llh_set
        self.solve_iter = solve_iter
        self.heuristics = LLHolder(self.llh_set)
        # 定义动作空间
        self.action_space = spaces.Discrete(len(self.heuristics))
        # 定义状态空间, 有 0-9 共 10 个整数状态
        # self.observation_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32)

        # 定义状态空间
        # self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,))
        # self.observation_space = spaces.Tuple((spaces.Discrete(len(self.heuristics)), spaces.Discrete(5000), spaces.Box(-float('inf'), float('inf'), shape=(1,))))

        # 配合前两版状态
        low = np.array([0.0, 0, -float('inf')])
        high = np.array([len(self.heuristics), self.solve_iter, float('inf')])

        #配合新版状态（cla那版）
        low = np.array([20, 0, -float('inf')])
        high = np.array([40, self.solve_iter, float('inf')])

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        #
        # self.observation_space = spaces.Discrete(10)
        # self.state = None
        # self.env_name = 'hh_env'  # the name of this env.
        self.prev_loss = 0

        # self.state_dim = self.observation_space.shape[0]  # feature number of state
        # self.action_dim = self.action_space.n  # feature number of action
        # self.if_discrete = True  # discrete action or continuous action

    def step(self, action):

        #action_c = np.argmax(action)
        action_c = action
        self.callCount[action_c] += 1
        self.ITER += 1
        newSolution = self.heuristics[action_c](self.solution, self.parameters)
        # prevTime = llh.timeTaken(self.solution, self.parameters)
        newTime = llh.timeTaken(newSolution, self.parameters)
        #termination = self.ITER > 5000 or newTime == self.TERMINATION_TIME
        termination = self.termination()

        # 奖励
        # reward = self.reward3B(newTime)
        reward = self.reward3D(newTime) #配合OI策略使用
        #print(" newTime: ", newTime)
        # 解的接受
        self.acceptOI(newTime, newSolution, action_c)

        #newstate2
        delta = self.prevTime - newTime


        if action_c in [3, 5, 6]:
            cla = 20

        else:
            cla = 40
        self.heuristicNo.append(action_c)
        self.heuristicType.append(cla)
        self.iter_type.append(self.ITER)
        self.step_result.append(self.bestTime)
        s_ = np.array([cla, self.NOT_IMPROVED, delta])

        if termination:
            self.render()
            # reward = self.endReward4()
            return s_, reward, termination, {'bestTime': self.bestTime}
        return s_, reward, termination, {}

    def accept(self, newTime, newSolution, action):
        if self.prevTime > newTime:
            self.improveCount[action] += 1
            self.solution = newSolution
            self.prevTime = newTime
            if (self.bestTime > newTime):
                self.best_solution = newSolution
                self.bestTime = newTime
                print("iter: ", self.ITER, "new bestTime: ", self.bestTime)
            self.NOT_ACCEPTED = 1
            self.NOT_IMPROVED = 1
        elif self.prevTime < newTime:
            self.NOT_IMPROVED += 1
            #print(' ')
            #print('ori prevTime: ', self.prevTime, 'newTime: ', newTime)
            # 解的接受
            p = random.random()
            #模拟退火
            temp = math.exp(-(newTime - self.prevTime) / (self.NOT_ACCEPTED *0.01))
            #print('NOT_IMPROVED: ', self.NOT_IMPROVED, 'temp: ', temp, 'p: ', p)
            if p < temp:
                #print('accepted!!!!!!!!!!!!!!!!!!!!!!!!!!')

                self.solution = newSolution
                self.prevTime = newTime
                #print('new prevTime: ', self.prevTime, 'newTime: ', newTime)
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

    def acceptOI(self, newTime, newSolution, action):
        if self.prevTime > newTime:
            self.improveCount[action] += 1
            self.solution = newSolution
            self.prevTime = newTime
            if (self.bestTime > newTime):
                self.best_solution = newSolution
                self.bestTime = newTime
                print("iter: ", self.ITER, "new bestTime: ", self.bestTime)
            self.NOT_ACCEPTED = 1
            self.NOT_IMPROVED = 1
        else:
            self.NOT_ACCEPTED += 1
            self.NOT_IMPROVED += 1
        # elif self.prevTime < newTime:
        #     self.NOT_IMPROVED += 1
        #     #print(' ')
        #     #print('ori prevTime: ', self.prevTime, 'newTime: ', newTime)
        #     # 解的接受
        #     p = random.random()
        #     #模拟退火
        #     temp = math.exp(-(newTime - self.prevTime) / (self.NOT_ACCEPTED *0.01))
        #     #print('NOT_IMPROVED: ', self.NOT_IMPROVED, 'temp: ', temp, 'p: ', p)
        #     if p < temp:
        #         #print('accepted!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #
        #         self.solution = newSolution
        #         self.prevTime = newTime
        #         #print('new prevTime: ', self.prevTime, 'newTime: ', newTime)
        #         self.NOT_ACCEPTED = 1
        #         self.NOT_IMPROVED += 1
        #     else:
        #         #print('declined!!!!!!!!!')
        #         self.NOT_ACCEPTED += 1
        #         self.NOT_IMPROVED += 1
        # else:
        #     self.solution = newSolution
        #     self.prevTime = newTime
        #     #print('new prevTime: ', self.prevTime, 'newTime: ', newTime)
        #     self.NOT_ACCEPTED += 1
        #     self.NOT_IMPROVED += 1
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

    def reward3B(self, newTime):
        if self.prevTime > newTime:
            reward = 10
            if self.bestTime > newTime:
                reward = 100
            self.rewardImp += reward
        elif self.prevTime == newTime:
            if self.NOT_IMPROVED <= 5:
                reward = 0
            else:
                reward = -1
            self.rewardSta += reward
        else:
            if self.NOT_IMPROVED <= 5:
                reward = -1
            else:
                reward = 0
            self.rewardMut += reward
        return reward

    def reward3D(self, newTime):
        if self.prevTime > newTime:
            reward = 10
            if self.bestTime > newTime:
                reward = 100
            self.rewardImp += reward
        else:
            reward = 0
        return reward
    def endReward4(self):
        #BestTime越小,奖励值越高
        if (self.bestTime in range(IDEAL_TIME[self.problem][0], IDEAL_TIME[self.problem][1] + 1)):
            finishRewardRate = 150 - (self.bestTime - IDEAL_TIME[self.problem][0]) / (IDEAL_TIME[self.problem][1] - IDEAL_TIME[self.problem][0] + 1) * 50
        else:
            finishRewardRate = 100 - (self.bestTime - IDEAL_TIME[self.problem][1]) / (IDEAL_TIME[self.problem][1] - IDEAL_TIME[self.problem][0] + 1) * 200
        print("end reward: ", finishRewardRate)
        return finishRewardRate

    def termination(self):
        #if self.bestTime in range(IDEAL_TIME[self.problem][0], IDEAL_TIME[self.problem][1]) or self.ITER > 5000:
        if self.bestTime <= IDEAL_TIME[self.problem][0] or self.ITER > self.solve_iter:
            return True
        else:
            return False
    def reset(self, **kwargs):
        self.TERMINATION_TIME = BEST_TIME[self.problem]
        self.parameters = parser.parse(self.problem_path)
        self.best_solution = self.solution = encoding.initializeFixedResult(self.parameters)
        self.NOT_ACCEPTED = 1
        self.NOT_IMPROVED = 1
        self.rewardImp = 0
        self.rewardMut = 0
        self.rewardSta = 0
        self.ITER = 1
        self.prevTime = self.bestTime = llh.timeTaken(self.solution, self.parameters)
        self.prevState = random.randint(0, 10)
        self.callCount = [0 for i in range(len(self.heuristics))]
        self.improveCount = [0 for i in range(len(self.heuristics))]
        # 记录采用的启发式类型
        self.heuristicType = []
        self.heuristicNo = []
        self.iter_type = []
        self.step_result = []
        # 返回一个一维的整型张量,随机取值,取值范围是[0,10)
        return np.array([self.prevState, self.NOT_IMPROVED, 0])

    def render(self, mode='human'):
        print('LLH called: ')
        print(self.callCount)
        print('improve called: ')
        print(self.improveCount)
        print('reward from improve: ', self.rewardImp)
        print('reward from stay: ', self.rewardSta)
        print('reward from mutation: ', self.rewardMut)
        print('total reward', self.rewardImp + self.rewardSta + self.rewardMut)
        print("finish time: ", self.bestTime)
        # print('iter: ', self.iter_type)
        # print('heuristic type: ', self.heuristicType)
        # print('heuristic no: ', self.heuristicNo)
        # gantt_data = decoding.translate_decoded_to_gantt(decoding.decode(self.parameters, self.best_solution[0], self.best_solution[1]))
        # gantt.draw_chart(gantt_data)
        # plt.title("MK06")
        # x = range(len(self.step_result))
        # plt.plot(x, self.step_result)  # o-:圆形
        # plt.xlabel("ITER")  # 横坐标名字
        # plt.ylabel("BestTime")  # 纵坐标名字
        # # plt.legend(loc="best")  # 图例
        # plt.show()

    def close(self):
        pass

    def test(self):
        print('problem', self.problem)
    #
    # def stepTest(self, action):
    #     newSolution = self.heuristics[action](self.solution, self.parameters)
    #     # prevTime = llh.timeTaken(self.solution, self.parameters)
    #     newTime = llh.timeTaken(newSolution, self.parameters)
    #     # 奖励函数
    #     if self.prevTime > newTime:
    #         self.solution = newSolution
    #         self.prevTime = newTime
    #         if (self.bestTime > newTime):
    #             self.best_solution = newSolution
    #             self.bestTime = newTime
    #
    #         self.NOT_ACCEPTED = 1
    #     else:
    #         # 解的接受
    #         p = random.random()
    #         temp = np.exp(-(newTime - self.prevTime) / (self.NOT_ACCEPTED * 0.01))
    #         if p < temp:
    #             print('accepted!')
    #             self.solution = newSolution
    #             self.prevTime = newTime
    #             self.NOT_ACCEPTED = 1
    #         else:
    #             self.NOT_ACCEPTED += 1
    #             print("NOT ACCEPTED count: ", self.NOT_ACCEPTED)
    #
    #     if action in [3, 5, 6]:
    #         ck = 20
    #     else:
    #         ck = 40
    #     s_ = (self.prevTime - newTime) / self.prevTime + ck
    #     s_ = np.array([s_], dtype=np.float32)
    #     return s_, 0, self.ITER > 3000, {}
    #
    # def reward(self, newTime):
    #     if self.prevTime > newTime:
    #         reward = 3 + self.NOT_IMPROVED * 0.01
    #         self.rewardImp += reward
    #         # print("imp ", self.rewardImp)
    #     else:
    #         if self.prevTime == newTime:
    #             reward = 0
    #         else:
    #             # reward = self.NOT_IMPROVED * 10 / self.ITER
    #             # reward = 2 * math.exp(-(35 / self.NOT_IMPROVED)) - 1
    #             reward = 0.1
    #             # if self.NOT_IMPROVED <= 8:
    #             #     reward = 0
    #             # else:
    #             #     reward = 0.005 * self.NOT_IMPROVED
    #             self.rewardMut += reward
    #             # reward = -1
    #             # print("mut reward: ", reward)
    #     return reward
    # def endReward(self):
    #     if (self.bestTime in range(IDEAL_TIME[self.problem][0], IDEAL_TIME[self.problem][1] + 1)):
    #         finishRewardRate = 200
    #     else:
    #         finishRewardRate = 100
    #     reward = (self.TERMINATION_TIME / self.bestTime) * finishRewardRate
    #     print("end reward: ", reward)
    #     return reward
    # def reward2(self, newTime):
    #     if self.prevTime > newTime:
    #         reward = -0.5
    #         self.rewardImp += reward
    #     else:
    #         if self.prevTime == newTime:
    #             reward = -2
    #             self.rewardSta += reward
    #         else:
    #             reward = -1
    #             self.rewardMut += reward
    #     return reward
    # def endReward2(self):
    #     if (self.bestTime in range(IDEAL_TIME[self.problem][0], IDEAL_TIME[self.problem][1] + 1)):
    #         finishRewardRate = 10000
    #     else:
    #         finishRewardRate = 5000
    #     reward = (self.TERMINATION_TIME / self.bestTime) * finishRewardRate
    #     print("end reward: ", reward)
    #     return reward
    # def endReward3(self):
    #     if (self.bestTime in range(IDEAL_TIME[self.problem][0], IDEAL_TIME[self.problem][1] + 1)):
    #         finishRewardRate = 200
    #     else:
    #         finishRewardRate = 100
    #     reward = (self.TERMINATION_TIME / self.bestTime) * finishRewardRate
    #     print("end reward: ", reward)
    #     return reward

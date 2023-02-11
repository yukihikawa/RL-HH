import gym
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from src.HLS.HHDQN_SS.env import hh_env
# 超参数
BATCH_SIZE = 32 # 批训练的数据个数
LR = 0.05 # 学习率
EPSILON = 0.9 # 贪婪度 greedy policy
GAMMA = 0.9 # 奖励递减值
TARGET_REPLACE_ITER = 100 # Q 现实网络的更新频率
MEMORY_CAPACITY = 4000 # 记忆库大小

myenv = gym.make('hh_env-v0')

N_ACTIONS = myenv.action_space.n # 获取动作的个数(10),输出维度
N_STATES = myenv.observation_space.shape[0] # 获取状态的个数(4),输入维度

print(N_ACTIONS)
print(N_STATES)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1) # initialization
        self.fc2 = nn.Linear(50, 50)
        self.fc2.weight.data.normal_(0, 0.1) # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1) # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2)) # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.p_loss = 0

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # x = torch.LongTensor(x)
        if np.random.uniform() < EPSILON:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] # return the argmax index
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    # 存储记忆
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a) # shape (batch, 1)
        q_next = self.target_net(b_s_).detach() # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        self.p_loss = loss


        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
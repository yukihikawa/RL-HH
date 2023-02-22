from src.HLS.DQN2_ERL_VNS.env.vns_env import hh_env


class Example:
    def __init__(self, value):
        self.value = value

    def print_value(self):
        print(self.value)

obj = Example(10)
obj.print_value()  # 输出 10

setattr(obj, 'value', 20)
obj.print_value()  # 输出 20

env = hh_env()
env.test()
setattr(env, 'problem', 20)
env.test()
setattr(env, 'problem', 30)
env.test()

import gym
env2 = gym.make('hh_env-v0')
print(env2.render())
env2.test()
setattr(env, 'problem', 20)
env2.test()
setattr(env, 'problem', 30)
env2.test()

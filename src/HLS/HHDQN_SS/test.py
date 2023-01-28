import torch
from src.HLH.HHDQN_SS import net_ss
import time
import config

dqn = net_ss.DQN()
dqn.eval_net.load_state_dict(torch.load('eval_model new net.pth'))

print(torch.cuda.is_available())

s = net_ss.myenv.reset()
t0 = time.time()
round = 0
count = [0 for i in range(10)]
for round in range(config.INNER_ITER):
    print("Round: %s" % round)
    # net.myenv.render()
    a = dqn.choose_action(s)
    # print('a: ', a)
    count[a] += 1
    s_, r, done, info = net_ss.myenv.step(a)

    round += 1

    s = s_  #
    print("Time: ", net_ss.myenv.prevTime)
net_ss.myenv.render()
t1 = time.time()
total_time = t1 - t0
print("Finished in {0:.2f}s".format(total_time))
print(count)
net_ss.myenv.render()

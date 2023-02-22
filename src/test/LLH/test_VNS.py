import random

from src.LLH.LLHSetVNS import LLHSetVNS
import os

problem = 'MK09'
problem_str = os.path.join(os.getcwd(), "../../Brandimarte_Data/" + problem + ".fjs")
set = LLHSetVNS()
# print(set.previous_time)
set.reset(problem_str)
# for i in range(0, 50):
#     # set.reset()
#     print('ori: ', set.previous_time)
#     for i in range(0, 100):
#         # 随机选取 0-4
#         idx = random.randint(0, 4)
#         # 执行对应的函数
#         set.llh[idx]()
#     print("local optimum: ", set.previous_time)
#     print('best: ', set.best_time)
#     # 随机选取 5-7
#     idx = random.randint(5, 7)
#     # 执行对应的函数
#     set.llh[idx]()
#     print("new: ", set.previous_time)
for i in range(0, 40):
    set.reset(problem_str)
    print('ori: ', set.previous_time)
    set.heuristicD()
    print("new: ", set.previous_time)
    print(' ')

from src.LLH.LLHSetVNS import LLHSetVNS
import os

problem = 'MK04'
problem_str = os.path.join(os.getcwd(), "../../Brandimarte_Data/" + problem + ".fjs")
set = LLHSetVNS(problem_str)
print(set.previous_time)
# for i in range(0, 40):
#     # set.reset()
#     print('ori: ', set.previous_time)
#     for i in range(0, 40):
#         set.heuristic3()
#         set.heuristic1()
#         set.heuristic2()
#     print("local optimum: ", set.previous_time)
#     print('best: ', set.best_time)
#     set.heuristicA()
#     print("new: ", set.previous_time)
for i in range(0, 40):
    set.reset()
    print('ori: ', set.previous_time)
    set.heuristic6()
    print("new: ", set.previous_time)
    print(' ')

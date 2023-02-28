import os

from src.LLH.LLHSetILS import LLHSetILS
from src.LLH.LLHUtils import timeTaken

PROBLEM = 'MK02'
PROBLEM_PATH = os.path.join(os.getcwd(), "../../Brandimarte_Data/" + PROBLEM + ".fjs")
N_MAX = 200
I_MAX = 1000

LLH_manager = LLHSetILS()
LLH_manager.reset(PROBLEM_PATH)

for i in range(I_MAX):
    n = 1
    l = 1
    best_solution = curr_solution = LLH_manager.previous_solution
    best_time = curr_time = LLH_manager.previous_time
    while n <= N_MAX:
        if l == 1:
            # 多重swap邻域,  Gmax= 3
            new_solution = LLH_manager.vnd14(LLH_manager.previous_solution)
            new_time = timeTaken(new_solution, LLH_manager.parameters)
        elif l == 2:
            # 作业对换
            new_solution = LLH_manager.vnd15(LLH_manager.previous_solution)
            new_time = timeTaken(new_solution, LLH_manager.parameters)
        elif l == 3:
            # 多重swap邻域,  Gmax= 5
            new_solution = LLH_manager.vnd14_1(LLH_manager.previous_solution)
            new_time = timeTaken(new_solution, LLH_manager.parameters)
        else:
            # 可加工机器
            new_solution = LLH_manager.vnd13(LLH_manager.previous_solution)
            new_time = timeTaken(new_solution, LLH_manager.parameters)

        if LLH_manager.previous_time > new_time:
            print('new Time: ', new_time)
            LLH_manager.accept_proposal_solution(new_solution, new_time)
        else:
            l += 1
        if l == 5:
            l = 1
        n += 1
print('best Time: ', LLH_manager.best_time)
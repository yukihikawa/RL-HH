import os
import random

from src.LLH.LLHSetILS import LLHSetILS
from src.LLH.LLHUtils import timeTaken

PROBLEM = 'MK09'
PROBLEM_PATH = os.path.join(os.getcwd(), "../../Brandimarte_Data/" + PROBLEM + ".fjs")
holder = LLHSetILS()
RENDER_TIMES = 20
ITER = 9000

results = []
for i in range(1, 20):

    holder.reset(PROBLEM_PATH)
    for j in range(ITER):
        prev_solution = holder.previous_solution
        prev_time = holder.previous_time
        idx = random.randint(0, len(holder.vnd) - 1)
        new_solution = holder.vnd[idx](prev_solution)
        new_time = timeTaken(new_solution, holder.parameters)
        if prev_time > new_time:
            holder.previous_solution = new_solution
            holder.previous_time = new_time
            print("Improved: ", holder.previous_time)
    print("Final: ", holder.previous_time)
    results.append(holder.previous_time)
print(results)
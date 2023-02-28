
from src.LLH.LLHUtils import timeTaken, get_machine_workload, timeTakenForDecoded
from src.LLH.LLHolder import LLHolder
import os

from src.utils import decoding, gantt
from src.utils.decoding import split_ms
from src.utils.encoding import initializeResult
from src.utils.parser import parse

problem = 'MK06'
problem_str = os.path.join(os.getcwd(), "../../Brandimarte_Data/" + problem + ".fjs")
holder = LLHolder(7)
llh = holder.set.llh[2]
llh2 = holder.set.llh[1]

parameters = parse(problem_str)
jobs = parameters['jobs']
# for i in range(len(jobs)):
#     print('job: ', i)
#     for op in jobs[i]:
#         print(op)

solution = initializeResult(parameters)
ms_s = split_ms(parameters, solution[1])
# for i in range(len(ms_s)):
#     print('job: ', i)
#     for op in range(len(ms_s[i])):
#         print('op: ', op+1, 'selected machine: ', jobs[i][op][ms_s[i][op]])

# print(solution)
decoded = decoding.decode(parameters, solution[0], solution[1])
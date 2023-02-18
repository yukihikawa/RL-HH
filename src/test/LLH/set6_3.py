
from src.LLH.LLHUtils import timeTaken
from src.LLH.LLHolder import LLHolder
import os

from src.utils.encoding import initializeResult
from src.utils.parser import parse

problem = 'MK06'
problem_str = os.path.join(os.getcwd(), "../../Brandimarte_Data/" + problem + ".fjs")
holder = LLHolder(7)
llh = holder.set.llh[2]
llh2 = holder.set.llh[1]

parameters = parse(problem_str)
solution = initializeResult(parameters)
print(solution)
prevTime = timeTaken(solution, parameters)
for i in range(1000):
    newSolution = llh(solution, parameters)
    newSolution = llh2(newSolution, parameters)
    print("new result",timeTaken(newSolution, parameters))
    solution = newSolution
    # print(solution)

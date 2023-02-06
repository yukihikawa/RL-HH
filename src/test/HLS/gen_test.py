import src.HLS.GEN.genetic.gen_main as gen_main
from src.HLS.GEN.genetic.config import *

def testTwenty(PROBLEM):
    problem_path = os.path.join(os.getcwd(), "../../Brandimarte_Data/" + PROBLEM + ".fjs")
    print('result for ', PROBLEM, ':')
    result = {}
    for i in range(20):
        result[i] = gen_main.runForTest(problem_path)
    print(result)

if __name__ == '__main__':
    testTwenty('MK02')





from src.HLS.GEN.genetic.gen_ops import *
from src.HLS.GEN.genetic.config import *
from src.utils.parser import parse

if __name__ == '__main__':
    for PROBLEM in PROBLEM_SET:
        problem_path = os.path.join(os.getcwd(), "../../../Brandimarte_Data/" + PROBLEM + ".fjs")
        parameters = parse(problem_path)


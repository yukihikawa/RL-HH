import os

POP_SIZE = 50
CHROM_LENGTH = 70
GEN_NUM = 150

PROBLEM_SET = ['MK01', 'MK02', 'MK03', 'MK04', 'MK05', 'MK06', 'MK07', 'MK08', 'MK09', 'MK10']
PROBLEM = 'MK05'
PROBLEM_PATH = os.path.join(os.getcwd(), "../../../Brandimarte_Data/" + PROBLEM + ".fjs")

P_MUT = 0.3

CROSS_TIMES = 5
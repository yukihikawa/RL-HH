from gen_test import test

PROBLEM = 'MK10'
POP_SIZE = 50
CHROM_LENGTH = 20
GEN_NUM = 50
P_MUT = 0.3
CROSS_TIMES = 5
TEST_ITER = 10
LLH_SET = 1

if __name__ == '__main__':
    test(TEST_ITER, PROBLEM, GEN_NUM, CHROM_LENGTH, POP_SIZE, CROSS_TIMES, P_MUT, LLH_SET)
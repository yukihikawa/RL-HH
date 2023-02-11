from gen_test import test

PROBLEM = 'MK07'
POP_SIZE = 50
CHROM_LENGTH = 50
GEN_NUM = 50
P_MUT = 0.3
CROSS_TIMES = 7
TEST_ITER = 10
LLH_SET = 12

if __name__ == '__main__':
    test(TEST_ITER, PROBLEM, GEN_NUM, CHROM_LENGTH, POP_SIZE, CROSS_TIMES, P_MUT, LLH_SET)
import random

from src.LLH.LLHolder import LLHolder
LLH_SET = 2
LENGTH = 5000
NUM = 2
def generate_and_save():
    #获取 llhset
    holder = LLHolder(LLH_SET)
    #生成长度为 length 的序列
    algorithm = []
    for i in range(LENGTH):
        algorithm.append(random.randint(0, len(holder.set.llh) - 1))
    #将序列保存到同目录的 txt文件中,文件名为 'LLH_SET_LENGTH'格式,若没有该文件则新建一个
    with open('LLH_SET' + str(LLH_SET) + '_LENGTH' + str(LENGTH) + '_' + str(NUM)+ '.txt', 'w') as f:
        f.write(str(algorithm))

if __name__ == '__main__':
    generate_and_save()
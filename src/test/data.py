#
import math

list = [261, 256, 260, 264, 258, 263, 257, 263, 259, 262, 259, 260, 265, 264, 259, 261, 258, 258, 253, 261]
# 计算均值
mean = sum(list) / len(list)
# 计算方差
var = sum([(i - mean) ** 2 for i in list]) / len(list)
# 计算标准差
std = math.sqrt(sum([(i - mean) ** 2 for i in list]) / len(list))
# 最大最小值
max = max(list)
min = min(list)
# 全部输出
print("均值：", mean, "方差：", var, "标准差：", std, "最大值：", max, "最小值：", min)
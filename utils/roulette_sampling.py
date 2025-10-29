import numpy as np

def roulette_sample(prob_dist):

    # 轮盘赌选择，list1为 list []
        # 产生 [0, 1) 的随机数
        r = np.random.rand()
        s = 0
        for i, v in enumerate(prob_dist):
            s += v
            if s > r:
                # 返回个体索引{0，1，2，3}
                return i
        return len(prob_dist) - 1


l1 = [0.1, 0.2, 0.3, 0.4]
result1 = []
# 执行1000次，记录每个个体的选择结果
for i in range(20):
    index = roulette_sample(l1)
    result1.append(index)
print('result1 =', result1)

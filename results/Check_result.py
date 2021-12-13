# -*- coding: utf-8 -*-
# @Time    : 2021/12/9 14:59
# @Author  : LIU YI


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
result = []
for i in range(5):
    result.append(np.load('naswot_hook_logdet_nasbench201_cifar10__none_0.05_1_True_128_1_1_{}_new.npy'.format(i)))


columns = ['a', 'b', 'c', 'd', 'e']
result = pd.DataFrame(result)
result = result.T
# top_num = []

# for i in range(1, result.shape[0]):
#     result_new = result[0:i]
#     top_list = []
#     for i in range(5):
#         # tep  = result[i]
#         ind = result_new[i].idxmax()
#         top_list.append(ind)
#     top_num.append(len(set(top_list)))

top_list = []
result = result[0:5]
for i in range(5):
    # tep  = result[i]
    ind = result[i].idxmax()
    top_list.append(ind)
print(top_list)
# for index in set(top_list):
#     print(result.)
# top_list = list(set(top_list))
# tep = result.loc[top_list]
# tep = tep.T
# print(tep)
# tep.plot(kind = 'bar')
# plt.ylim(1740, 1755)
# plt.show()
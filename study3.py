# 正规方程

import numpy as np
import pandas as ps

path = 'resource/all_work_code/exerise1/ex1data2.txt'
data = ps.read_csv(path, header=None, names=['size', 'homes', 'prices'])
data = (data - data.mean()) / data.std()
data.insert(0, "ones", 1)
cols = data.shape[1]
print(cols)
x = data.iloc[:, 0:(cols - 1)]
y = data.iloc[:, (cols - 1):cols]


def regularDefensePort(x, y):
    theta = np.linalg.inv(x.T @ x) @ x.T @ y
    return theta


theta = regularDefensePort(x, y)
print(theta)

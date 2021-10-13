import numpy as np
import pandas as ps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load data
path = 'resource/all_work_code/exerise1/ex1data2.txt'
data = ps.read_csv(path, header=None, names=['size', 'homes', 'prices'])
# print(data.head())

# 对数据进行处理, 特征归一化
# 每一维减去均值,除以方差
data = (data - data.mean()) / data.std()

data.insert(0, 'one', 1)
print(data.head())

# 绘图3D
fig = plt.figure()
ax = Axes3D(fig)
xx = data.iloc[:, 1:2]
yy = data.iloc[:, 2:3]
zz = data.iloc[:, 3:4]
ax.scatter(xx.values, yy.values, zz.values, c='b', marker='*')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

# init x y theta
cols = data.shape[1]
print(cols)
x = data.iloc[:, 0:(cols - 1)]
y = data.iloc[:, (cols - 1):cols]

x = np.matrix(x.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0, 0]))


# 梯度下降法

# 梯度下降, x,y theta初始猜测值,alpha学习速率,iters学习次数
def grandientDecline(x, y, theta, alpha, iters):
    tmp = np.matrix(np.zeros(theta.shape))
    # 求解的参数个数
    parameters = int(theta.ravel().shape[1])
    # 构建学习次数个数组
    cost = np.zeros(iters)

    for i in range(iters):
        error = (x * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, x[:, j])
            tmp[0, j] = theta[0, j] - ((alpha / len(x)) * np.sum(term))

        theta = tmp
        cost[i] = computeCost(x, y, theta)

    return theta, cost


# 计算平均误差
def computeCost(x, y, theta):
    inner = np.power(((x * theta.T) - y), 2)
    return (1 / (2 * len(x))) * np.sum(inner)


g, cost = grandientDecline(x, y, theta, 0.01, 1000)
print(g)

# 绘制收敛函数
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(1000), cost, 'r')
ax.set_xlabel('iterations')
ax.set_ylabel('cost')
ax.set_title('error')
plt.show()

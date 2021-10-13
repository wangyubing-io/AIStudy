import matplotlib.pyplot as plt
import numpy as np
import pandas as ps

# load data 人口与收入的关系
path = 'resource/all_work_code/exerise1/ex1data1.txt'
data = ps.read_csv(path, header=None, names=['Population', 'Profit'])
# print(data.head())

# 绘图
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
plt.show()

data.insert(0, 'Ones', 1)

# init x y theta
cols = data.shape[1]
# 第一列和第二列
x = data.iloc[:, 0:cols - 1]
# print(x.head())
# 第三列,输出
y = data.iloc[:, cols - 1:cols]
# print(y.head())
# init matrix
x = np.matrix(x.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))


# 计算平均误差
def computeCost(x, y, theta):
    inner = np.power(((x * theta.T) - y), 2)
    return (1 / (2 * len(x))) * np.sum(inner)


print('误差=', computeCost(x, y, theta))


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


alpha = 0.01
iters = 1000

g, cost = grandientDecline(x, y, theta, alpha, iters)
print(type(g))
print(g)

print("新误差=", computeCost(x, y, g))

# 绘制函数图 100个样本
x = np.linspace(data.Population.min(), data.Population.max(), 100)
# f = theta0 + theta1 * x
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=4)  # 显示标签位置
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Prediction Profit vs Population Size')
plt.show()

# 绘制收敛数据图
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('iter')
ax.set_title('Error cs . Training Epoch')
plt.show()

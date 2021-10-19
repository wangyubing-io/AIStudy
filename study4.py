import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report  # 评估报告

plt.style.use('fivethirtyeight')

# 逻辑回归, 分类

# 准备数据
data = pd.read_csv('resource/all_work_code/exerise2/ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
print(data.head())

# 数据可视化
sns.set(context='notebook', style='darkgrid', palette=sns.color_palette('RdBu', 2))
sns.lmplot('exam1', 'exam2', hue='admitted', data=data, size=6, fit_reg=False, scatter_kws={'s': 50})
plt.show()


# 特征缩放
def normalize_feature(data):
    data.apply(lambda column: (column - column.mean()) / column.std())


# 获取x
def getX(data):
    # 生成一个数据长度的zero向量
    ones = pd.DataFrame({'ones': np.zeros(len(data))})
    # 连接到data
    data = pd.concat([ones, data], axis=1)
    # 裁切特征
    return data.iloc[:, :-1].values


# 获取y
def getY(data):
    return data.iloc[:, -1].values


X = getX(data)
y = getY(data)

print(X.shape)
print(y.shape)


# sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# g(z) yu h(x)

# 绘制
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.arange(-10, 10, step=0.01),
        sigmoid(np.arange(-10, 10, step=0.01)))
ax.set_ylim((-0.1, 1.1))  # lim 轴线显示长度
ax.set_xlabel('z', fontsize=18)
ax.set_ylabel('g(z)', fontsize=18)
ax.set_title('sigmoid', fontsize=18)
plt.show()

#

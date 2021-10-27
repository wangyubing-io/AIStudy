import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report  # 评估报告
import study4

plt.style.use('fivethirtyeight')

# 逻辑回归, 分类, 正则

# 准备数据
df = pd.read_csv('resource/all_work_code/exerise2/ex2data2.txt', names=['test1', 'test2', 'accepted'])
print(df.head())

# 数据可视化
sns.set(context='notebook', style='darkgrid', palette=sns.color_palette('RdBu', 2))
sns.lmplot('test1', 'test2', hue='accepted', data=df, size=6, fit_reg=False, scatter_kws={'s': 50})
plt.show()


# TODO 特征映射 对特征进行扩展
def feature_mapping(x, y, power, as_ndarray=False):
    data = {'f{}{}'.format(i - p, p): np.power(x, i - p) * np.power(y, p)
            for i in np.arange(power + 1)
            for p in np.arange(i + 1)
            }
    if as_ndarray:
        return pd.DataFrame(data).values
    else:
        return pd.DataFrame(data)


x1 = np.array(df.test1)
x2 = np.array(df.test2)

X = feature_mapping(x1, x2, power=6, as_ndarray=True)
print(X.shape)

y = df.iloc[:, -1].values
print(y.shape)
# 正则化代价函数
theta = np.zeros(X.shape[1])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# def cost(theta, x, y):
#     return (1 / len(x)) / (-y * np.log(sigmoid(x @ theta)) - ((1 - y) * np.log(1 - sigmoid(x @ theta))))


def regularized_cost(theta, X, y, l=1):
    theta_j1_to_n = theta[1:]
    regularized_term = (l / (2 * len(X))) * np.power(theta_j1_to_n, 2).sum()
    return study4.cost(theta, X, y) + regularized_term


print(regularized_cost(theta, X, y, l=1))


# 正则梯度
def regularized_gradient(theta, X, y, l=1):
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n
    regularized_term = np.concatenate([np.array([0]), regularized_theta])
    return study4.gradient(theta, X, y) + regularized_term


print(regularized_gradient(theta, X, y, 1))

# 拟合 theta
import scipy.optimize as opt

print('init cost={}'.format(regularized_cost(theta, X, y, 1)))

res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y), method='Newton-CG', jac=regularized_gradient)
print(res)


def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)


# 预估
final_theta = res.x

y_pred = predict(X, final_theta)
print(classification_report(y, y_pred))


# 决策边界
def draw_boundary(power, l):
    density = 1000
    threshhold = 2 * 10 ** -3

    final_theta = feature_mappe_logistic_regression(power, l)
    x, y = find_decision_bounday(density, power, final_theta, threshhold)
    df = pd.read_csv('resource/all_work_code/exerise2/ex2data2.txt', names=['test1', 'test2', 'accepted'])
    sns.lmplot('test1', 'test2', hue="accepted", data=df, size=6, fit_reg=False, scatter_kws={'s': 100})
    plt.scatter(x, y, c='R', s=10)
    plt.title('noundary')
    plt.show()


def feature_mappe_logistic_regression(power, l):
    df = pd.read_csv('resource/all_work_code/exerise2/ex2data2.txt', names=['test1', 'test2', 'accepted'])
    x1 = np.array(df.test1)
    x2 = np.array(df.test2)
    y = np.array(df.accepted)

    X = feature_mapping(x1, x2, power, as_ndarray=True)
    theta = np.zeros(X.shape[1])
    res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y, l), method='TNC', jac=regularized_gradient)
    final_theta = res.x
    return final_theta


def find_decision_bounday(density, power, theta, threshhold):
    t1 = np.linspace(-1, 1.5, density)
    t2 = np.linspace(-1, 1.5, density)
    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    mapped_card = feature_mapping(x_cord, y_cord, power)
    inner_product = mapped_card.as_matrix() @ theta
    decision = mapped_card[np.abs(inner_product) < threshhold]
    return decision.f10, decision.f01

draw_boundary(power=6, l=1)
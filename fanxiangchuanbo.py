# 反向传播
import numpy as np

# 初始化数据
i1 = 0.05
i2 = 0.10

o1 = 0.01
o2 = 0.99

# 目标：给出输入数据i1,i2(0.05和0.10)，使输出尽可能与原始输出o1,o2(0.01和0.99)接近。

w1 = 0.15
w2 = 0.20
w3 = 0.25
w4 = 0.30
w5 = 0.40
w6 = 0.45
w7 = 0.50
w8 = 0.55

# 扩展列
b1 = 0.35
b2 = 0.60

# 学习速率
study_rate = 0.5


# sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 向前传播
net_h1 = w1 * i1 + w2 * i2 + b1
net_h2 = w3 * i1 + w4 * i2 + b1

out_h1 = sigmoid(net_h1)
out_h2 = sigmoid(net_h2)

net_o1 = w5 * out_h1 + w6 * out_h2 + b2
net_o2 = w7 * out_h1 + w8 * out_h2 + b2

out_o1 = sigmoid(net_o1)
out_o2 = sigmoid(net_o2)

# 初始化后的输出值
print("o1={},o2={}".format(out_o1, out_o2))

# 总误差函数
e_o1 = (1 / 2) * np.power((o1 - out_o1), 2)
e_o2 = (1 / 2) * np.power((o2 - out_o2), 2)
e_total = e_o1 + e_o2


def e_total():
    e_o1 = (1 / 2) * np.power((o1 - out_o1), 2)
    e_o2 = (1 / 2) * np.power((o2 - out_o2), 2)
    return e_o1 + e_o2


print("e_o1={},e_o2={},e_total={}".format(e_o1, e_o2, e_total))


# 输出层误差
def e_out(target_out, out_o0):
    return - (target_out - out_o0)


# 反向传播
def e_total_w5678(target, out, out_h):
    return -(target - out) * out * (1 - out) * out_h


def e_total_w1234(target_o1, out_o1, w1, target_o2, out_o2, w2, out_h, i):
    return (e_out(target_o1, out_o1) * out_o1 * (1 - out_o1) * w1 + (
            e_out(target_o2, out_o2) * out_o2 * (1 - out_o2) * w2)) * out_h * i


for i in range(10000):
    # 更新
    w1 = w1 - (study_rate * e_total_w1234(o1, out_o1, w5, o2, out_o2, w7, out_h1, i1))
    w2 = w2 - (study_rate * e_total_w1234(o1, out_o1, w5, o2, out_o2, w7, out_h1, i2))
    w3 = w3 - (study_rate * e_total_w1234(o1, out_o1, w6, o2, out_o2, w8, out_h2, i1))
    w4 = w4 - (study_rate * e_total_w1234(o1, out_o1, w6, o2, out_o2, w8, out_h2, i2))

    w5 = w5 - (study_rate * e_total_w5678(o1, out_o1, out_h1))
    w6 = w6 - (study_rate * e_total_w5678(o1, out_o1, out_h2))
    w7 = w7 - (study_rate * e_total_w5678(o2, out_o2, out_h1))
    w8 = w8 - (study_rate * e_total_w5678(o2, out_o2, out_h2))

    # 更新
    net_h1 = w1 * i1 + w2 * i2 + b1
    net_h2 = w3 * i1 + w4 * i2 + b1

    out_h1 = sigmoid(net_h1)
    out_h2 = sigmoid(net_h2)

    net_o1 = w5 * out_h1 + w6 * out_h2 + b2
    net_o2 = w7 * out_h1 + w8 * out_h2 + b2

    out_o1 = sigmoid(net_o1)
    out_o2 = sigmoid(net_o2)

    if (i % 1000 == 0):
        print("第{}次更新后的误差为{}".format(i, e_total()))

print("z最终输出out_o1={},out_o2={}".format(out_o1, out_o2))
print("目标输出target1={},target2={}".format(o1, o2))
print("误差值为e1={},e2={}".format(o1 - out_o1, o2 - out_o2))

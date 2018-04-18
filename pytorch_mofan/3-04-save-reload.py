# -*- coding:utf-8 -*-

"""
https://morvanzhou.github.io/tutorials/machine-learning/torch/3-01-regression/
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
x, y = Variable(x), Variable(y) # 用 Variable 来修饰这些数据 tensor



def CreateNet():
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    # optimizer 是训练的工具
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率
    loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

    for t in range(100):
        prediction = net.forward(x)     # 喂给 net 训练数据 x, 输出预测值
        loss = loss_func(prediction, y)     # 计算两者的误差
        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        loss.backward()         # 误差反向传播, 计算参数更新值
        optimizer.step()        # 将参数更新值施加到 net 的 parameters 上

    Paint(131, x, prediction)

    # 保存神经网络
    torch.save(net, "net.pkl")  # 存整个net
    torch.save(net.state_dict(), "net_params.pkl") # 只保存网络中的参数


# 提取神经网络
def LoadAll():
    net2 = torch.load("net.pkl")
    prediction = net2(x)
    Paint(132, x, prediction)

def LoadParams():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net3.load_state_dict(torch.load("net_params.pkl"))
    prediction = net3(x)
    Paint(133, x, prediction)


def Paint(iNum, x, prediction):
    plt.subplot(iNum)
    plt.title(str(iNum))
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'g-', lw=5)


plt.figure(1, figsize=(10, 3))
CreateNet()
LoadAll()
LoadParams()
plt.show()

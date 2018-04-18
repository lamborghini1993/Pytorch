# -*- coding:utf-8 -*-

"""
https://morvanzhou.github.io/tutorials/machine-learning/torch/3-02-classification/
"""

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable

n_data = torch.ones(100, 2)

x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), 0).type(torch.LongTensor)

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1],c=y.data.numpy(), s=100)
# plt.show()

## method 1
# class Net(torch.nn.Module):
#     def __init__(self, n_input, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_input, n_hidden)
#         self.predict = torch.nn.Linear(n_hidden, n_output)

#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.predict(x)
#         return x

# net = Net(2, 10, 2)

## method 2
net = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)


optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()
plt.show()

for t in range(100):
    out = net.forward(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.cla()
    t1 = F.softmax(out)
    t2 = torch.max(t1, 1)
    prediction = t2[1]
    pred_y = prediction.data.numpy().squeeze()
    target_y = y.data.numpy()
    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100)
    accuracy = sum(pred_y == target_y)/200  # 预测中有多少和真实值一样
    plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
    plt.pause(0.1)

plt.ioff()
plt.show()


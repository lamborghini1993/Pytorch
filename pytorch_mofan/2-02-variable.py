# -*- coding:utf-8 -*-
"""
变量 (Variable)
https://morvanzhou.github.io/tutorials/machine-learning/torch/2-02-variable/
rch 中的 Variable 就是一个存放会变化的值的地理位置. 里面的值会不停的变化. 就像一个裝鸡蛋的篮子, 鸡蛋数会不停变动. 那谁是里面的鸡蛋呢, 自然就是 Torch 的 Tensor 咯. 如果用一个 Variable 进行计算, 那返回的也是一个同类型的 Variable.
"""

import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)

t_out = torch.mean(tensor * tensor)
v_out = torch.mean(variable * variable)

v_out.backward()
print(variable.grad)
print("------------")
print(variable)
print(variable.data)
print(variable.data.numpy())

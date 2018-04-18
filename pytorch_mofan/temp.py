# -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable

m = torch.nn.Linear(2, 4)
# for x in dir(m):
#     print("{}\t:{}".format(x, getattr(m, x, "none")))
# print(dir(m))
print(m.__dict__)
t = torch.randn(5, 2)
# print(t)
input = Variable(t)
print(input)
output = m(input)
print(output)

"""
 0.3188  0.3094
 0.0828  0.3452
 0.4695 -0.3138
"""
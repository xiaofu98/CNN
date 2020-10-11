#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/10/11 9:05
# @Author  : Mr.X
# @File    : train.py
# @Software: PyCharm
import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

data_train = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=True)

data_test = datasets.MNIST(root="./data/",
                           transform=transform,
                           train=False)
data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=64, shuffle=True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=64, shuffle=True)

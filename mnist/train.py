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
from model import Model
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

model = Model()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

n_epochs = 5

for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-" * 10)

    for data in data_loader_train:
        X_train, y_train = data
        X_train, y_train = Variable(X_train), Variable(y_train)
        outputs = model(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)

        loss.backward()
        optimizer.step()
        running_loss += loss.data
        running_correct += torch.sum(pred == y_train.data)

    testing_correct = 0
    for data in data_loader_test:
        X_test, y_test = data
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)
    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss // len(data_train),
                                                                                      100 * running_correct // len(data_train),
                                                                                      100 * testing_correct // len(data_test)))

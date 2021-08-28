# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:40:05 2020

@author: lenovo
"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch
from torch.optim.lr_scheduler import MultiStepLR

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.fc1 = nn.Linear(32 * 32 * 3, 32 * 32 * 2)
        self.fc2 = nn.Linear(32 * 32 * 2, 32 * 32)
        self.fc3 = nn.Linear(32 * 32, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



net = LeNet()
net = net.cuda()
net.load_state_dict(torch.load('mod_CIFAR.pt'))
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
train_set = torchvision.datasets.CIFAR10(root="./data",
                                         train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=1000, shuffle=True, num_workers=0)
test_set = torchvision.datasets.CIFAR10(root="./data",
                                        train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=1000, shuffle=False, num_workers=0)
test_acc=0
for i, data in enumerate(test_loader):
        net.zero_grad()
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        inputs = inputs.view(-1, 32 * 32 * 3)
        output = net(inputs)
        _, pred = output.max(1)
        num_correct = (pred == labels).sum()
        test_acc += int(num_correct)
print(test_acc)


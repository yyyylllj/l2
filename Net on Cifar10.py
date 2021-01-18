# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 09:40:25 2020

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:46:34 2020

@author: lenovo
"""

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
BC=64
XLBC=0.01
Ln=0.5
LS=200
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(32 * 32 * 3, 32 * 32 * 2)
        self.fc2 = nn.Linear(32 * 32 * 2, 32 * 32)
        self.fc3 = nn.Linear(32 * 32, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def func(a,b):
    if torch.norm(a)>b:
        c=a/torch.norm(a)*b
        return(c)
    else:
        c=a
        return(c)
net = Net()
net = net.cuda()
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
train_set = torchvision.datasets.CIFAR10(root="./data",
                                         train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=BC, shuffle=True, num_workers=2)
test_set = torchvision.datasets.CIFAR10(root="./data",
                                        train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=BC, shuffle=False, num_workers=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=XLBC)
params=list(net.parameters())
for epoch in range(LS):
    for i, data in enumerate(train_loader):
        net.zero_grad()
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        inputs = inputs.view(-1, 32 * 32 * 3)
        output = net(inputs)
        train_loss = criterion(output, labels)
        train_loss.backward()
        optimizer.step()
        for ll in range(32*32*2):
            with torch.no_grad():
                n=params[0][ll,:]
                params[0][ll,:]=func(n,Ln)
        for ll in range(32*32):
            with torch.no_grad():
                n=params[2][ll,:]
                params[2][ll,:]=func(n,Ln)
#torch.save(net.state_dict(),'')
test_acc=0
for x,y in test_loader:
    x=x.view(-1,32*32*3)
    x=x.cuda()
    y=y.cuda()    
    out=net(x)
    _, pred = out.max(1)
    num_correct = (pred == y).sum()
    test_acc += int(num_correct)
print(test_acc/10000)

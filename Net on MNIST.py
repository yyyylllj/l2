# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:32:15 2021

@author: ylj和xqq
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
BC=64
XLBC=0.03
Ln=0.3
LS=200
train_dataset = datasets.MNIST(root='./data/',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root='./data/',train=False,transform=transforms.ToTensor(),download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BC,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BC,
                                          shuffle=False)
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.l11=nn.Linear(784,784)
        self.l2=nn.Linear(784,784)
        self.linear2=nn.Linear(784,10)
    def forward(self,x):
        x=F.relu(self.l11(x))
        x=F.relu(self.l2(x))
        x=self.linear2(x)
        return x
mod=Model()
mod=mod.cuda()
loss=nn.CrossEntropyLoss()
optimizer=optim.SGD(mod.parameters(),lr=XLBC)
losses=[]
acces=[]
eval_losses=[]
eval_acces=[]
def func(a,b):
    if torch.norm(a)>b:
        c=a/torch.norm(a)*b
        return(c)
    else:
        c=a
        return(c)
params=list(mod.parameters())
for i in range(LS):
    train_loss=0
    train_acc=0
    mod.train()
    for j,(X,label) in enumerate(train_loader):
        X=X.view(-1,784)
        X=Variable(X)
        label=Variable(label)
        X=X.cuda()
        label=label.cuda()
        out=mod(X)
        lossvalue=loss(out,label)
        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step()
        for ll in range(784):
            with torch.no_grad():
                n=params[0][ll,:]
                params[0][ll,:]=func(n,Ln)
                m=params[2][ll,:]
                params[2][ll,:]=func(m,Ln)   
    test_acc=0
    mod.eval()
    for x,y in test_loader:      
           x=x.view(-1,784)
           x=x.cuda()
           y=y.cuda()
           out=mod(x)
           _,pred=out.max(1)
           num_correct=(pred==y).sum()
           acc=int(num_correct)
           test_acc+=acc
    print(test_acc/10000)
#torch.save(mod.state_dict(),'mod_MNIST.pt')
#noise=torch.load('noise_test.pth')
#noisetest_loader = torch.utils.data.DataLoader(dataset=noise,batch_size=BC, shuffle=False)
#noisetest_acc=0
#for x,y in noisetest_loader:      
#  x=x.view(-1,784)
#  x=x.cuda()
#  y=y.cuda()
#  out=mod(x)
#   _,pred=out.max(1)
#  num_correct=(pred==y).sum()
#  acc=int(num_correct)
#  noisetest_acc+=acc
#print(noisetest_acc/10000)

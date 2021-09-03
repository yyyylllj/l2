import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
BC=100
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
mod.load_state_dict(torch.load('mod_MNIST.pt'))
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
print("accuracy:" + ' ' + str(test_acc / 10000))

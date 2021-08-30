import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
test_dataset = datasets.MNIST(root='./data/',train=False,transform=transforms.ToTensor(),download=True)
noise=[]
L_n=0.1
for x,y in test_dataset:
    mc=torch.randn(1,28,28)
    x+=mc*L_n
    noise.append((x,y))
#torch.save(noise,'noise_test.pth')

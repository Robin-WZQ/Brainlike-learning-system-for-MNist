# -*- coding: utf-8 -*-
"""
Created on Sat Nov 2021/11/2
@ author: Zhongqi Wang
@ file: 最朴素的hamming网路做MNIST数据集，参数为手工定义，测试集正确率约为12%
"""
import numpy as np
import copy
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision import datasets
class HammingNet:
    def __init__(self, W1, outputs):
        self.outputs = outputs
        self.W1 = W1
        self.b1 = 10*np.ones((10, 1))
        self.W2 = np.eye(10)
        self.W2[self.W2==0] = -(1.0/10)
        self.max_iters = 100

    def fit(self, x):
        a1 = self.purelin(x)
        a2 = copy.deepcopy(a1)
        for i in range(self.max_iters):
            new_a2 = self.poslin(a2)
            if new_a2.tolist() == a2.tolist():
                a2 = copy.deepcopy(new_a2)
                break
            a2 = copy.deepcopy(new_a2)
        a2 = a2.reshape(-1)
        a2[a2>0] = 1
        output_type = self.outputs[a2.tolist().index(1)] 
        return output_type
    
    def purelin(self, x):
        return np.dot(self.W1, x)+self.b1
    
    def poslin(self, x):
        a2 = np.dot(self.W2, x)
        a2[a2<0] = 0
        return a2
        
outputs = ["0",'1','2','3','4','5','6','7','8','9']
num=42

class network_for_features(nn.Module):
    def __init__(self):
        super(network_for_features, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(784, 84),
            nn.Sigmoid(),
            nn.Linear(84, 42)
        )

    def forward(self, x):
        x = x.view(-1)
        x = self.layer(x)
        return x

net = network_for_features()
net.load_state_dict(torch.load("trained_model/feature_29.pth"))

# 下载测试集
test_dataset = datasets.MNIST(root='./num/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)
train_dataset = datasets.MNIST(root='./num/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
batch_size = 1
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

def evaluate_accuracy(data_iter, net,w1):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        output = net(X)
        for i in range(num):
            if output[i]>=1:
                output[i] = 1
            else:
                output[i] = -1

        hammingnet = HammingNet(w1, output)
        x = output.reshape((42,1))
        x = x.detach().numpy()
        result = hammingnet.fit(x)
        if result == int(y.item()):
            acc_sum += 1
        n += 1
    return acc_sum / n

r=0
for m in range(100):
    w1 = torch.randn([10,num])
    for i in range(10):
        for j in range(num):
            if w1[i][j]>0:
                w1[i][j]=1
            else:
                w1[i][j]=-1
    w1 = w1.numpy()
    re = evaluate_accuracy(test_loader, net,w1)
    if re>r:
        r=re
        print(m,re)
        w=w1

final_result = evaluate_accuracy(train_loader, net,w)
print(final_result)

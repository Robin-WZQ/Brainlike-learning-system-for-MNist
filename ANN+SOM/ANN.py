# -*- coding: utf-8 -*-
"""
Created on Sat Nov 2021/11/6
@ author: Zhongqi Wang
@ file: ANN 网络
"""
import torch
from torch import nn
from torch.nn import init
import torchvision.transforms as transforms
from torchvision import datasets


# 下载训练集
train_dataset = datasets.MNIST(root='./num/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
# 下载测试集
test_dataset = datasets.MNIST(root='./num/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)
batch_size = 64
# 装载训练集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
# 装载测试集
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

class network_for_features(nn.Module):
    def __init__(self):
        super(network_for_features, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(784, 84),
            nn.Sigmoid(),
            nn.Linear(84, 42)
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.layer(x)
        return x

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(42,10)
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.layer(x)
        return x

net = network_for_features()
my_network = classifier()

from tensorboardX import SummaryWriter

writer = SummaryWriter('log99')

for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
    if 'bias' in name:
        init.constant_(param, val=0)

# 损失函数使用交叉熵
criterion = nn.CrossEntropyLoss()
# 优化函数使用 SGD
optimizer = torch.optim.SGD(net.parameters(), lr=0.04)
optimizer2 = torch.optim.SGD(my_network.parameters(), lr=0.04)

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        output = my_network(net(X))
        acc_sum += (output.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def query(data,net):
    output = net(data).argmax(dim=1)
    print(output)

num_epochs = 2
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,params=None, lr=None, optimizer=None,optimizer2=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m = 0.0, 0.0, 0, 0
        for X, y in train_iter:
            y_hat = my_network(net(X))
            l = loss(y_hat, y)
            
            # 梯度清零
            optimizer2.zero_grad()
            optimizer.zero_grad()
            
            l.backward() # 计算梯度

            optimizer.step()  # 随机梯度下降算法, 更新参数
            optimizer2.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            m += 1

            q=train_l_sum / m
            if epoch == 4:
                writer.add_scalar('Train/Loss', q , m)
                
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'% (epoch + 1, train_l_sum / m, train_acc_sum / n, test_acc))

        torch.save(net.state_dict(),"trained_model/feature_%d.pth" % (epoch))

if __name__ == "__main__":
    train_ch3(net, train_loader, test_loader, criterion, num_epochs, batch_size, None, None, optimizer,optimizer2)

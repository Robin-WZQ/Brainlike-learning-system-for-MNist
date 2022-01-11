# -*- coding: utf-8 -*-
"""
Created on Sat Nov 2021/11/6
@ author: Zhongqi Wang
@ file: 自组织特征映射网络
"""

# ==================================================  写在最前面——它是怎么工作的 ==================================================
#     首先，我们假设正如人识别物体从特征开始，对于网络的训练没有必要完全从0开始进行训练，可以将已经提取好的特征输入网络训练，也就是
# 说我们先用ANN网络训练出N维的向量（大于分类结果，此处是将数字映射为42维的特征，舍弃了最后一层的分类器）。同时为更好模拟人类神经元，
# 我们使用了SOM模型用作训练当作在生物神经系统中，存在着一种侧抑制现象，即一个神经细胞兴奋以后，会对周围其他神经细胞产生抑制作用。这
# 种抑制作用会使神经细胞之间出现竞争，其结果是某些获胜，而另一些则失败。表现形式是获胜神经细胞兴奋，失败神经细胞抑制。
#     特别的，由于最后10分类结果无法匹配上原label（就是说虽然分出10类但无法对应上），设计了转换字典，使每次分类结果能和label匹配。此
# 外，为了使模型具有可移植性，我将模型中的参数矩阵和转换字典保存起来，方便直接调用，而无需再次训练。最后达到了比肩CNN的效果，其中训练
# 集正确率达到94.26%，测试集正确率达到91.24%。关于模型的更多实验将在报告中详细讨论，这里不加赘述。
# ==================================================  写在最前面——它是怎么工作的 ==================================================
from torch.autograd import Variable
import json

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets

from capsnet import CapsNet


class Config:
  def __init__(self, dataset='mnist'):
    if dataset == 'mnist':
      # CNN (cnn)
      self.cnn_in_channels = 1
      self.cnn_out_channels = 256
      self.cnn_kernel_size = 9

      # Primary Capsule (pc)
      self.pc_num_capsules = 8
      self.pc_in_channels = 256
      self.pc_out_channels = 32
      self.pc_kernel_size = 9
      self.pc_num_routes = 32 * 6 * 6

      # Digit Capsule (dc)
      self.dc_num_capsules = 10
      self.dc_num_routes = 32 * 6 * 6
      self.dc_in_channels = 8
      self.dc_out_channels = 16

      # Decoder
      self.input_width = 28
      self.input_height = 28

    elif dataset == 'cifar10':
      # CNN (cnn)
      self.cnn_in_channels = 3
      self.cnn_out_channels = 256
      self.cnn_kernel_size = 9

      # Primary Capsule (pc)
      self.pc_num_capsules = 8
      self.pc_in_channels = 256
      self.pc_out_channels = 32
      self.pc_kernel_size = 9
      self.pc_num_routes = 32 * 8 * 8

      # Digit Capsule (dc)
      self.dc_num_capsules = 10
      self.dc_num_routes = 32 * 8 * 8
      self.dc_in_channels = 8
      self.dc_out_channels = 16

      # Decoder
      self.input_width = 32
      self.input_height = 32

    elif dataset == 'your own dataset':
      pass


# 自组织特征映射网络
class SOM(object):
  def __init__(self, X, output, iteration, batch_size):
    """
    :param X: 形状是N*D,输入样本有N个,每个D维
    :param output: (n,m)一个元组，为输出层的形状是一个n*m的二维矩阵
    :param iteration:迭代次数
    :param batch_size:每次迭代时的样本数量
    初始化一个权值矩阵，形状为D*(n*m)，即有n*m权值向量，每个D维
    """
    self.X = X
    self.output = output
    self.iteration = iteration
    self.batch_size = batch_size
    self.W = np.random.rand(X.shape[1], output[0] * output[1])
 
  def GetN(self, t):
    """
    :param t:时间t, 这里用迭代次数来表示时间
    :return: 返回一个整数，表示拓扑距离，时间越大，拓扑邻域越小
    """
    a = min(self.output)
    return int(a-float(a)*t/self.iteration)
 
  #求学习率
  def Geteta(self, t, n):
    """
    :param t: 时间t, 这里用迭代次数来表示时间
    :param n: 拓扑距离
    :return: 返回学习率，
    """
    return np.power(np.e, -n)/(t+2)
 
  #更新权值矩阵 
  def updata_W(self, X, t, winner):
    N = self.GetN(t)   #表示随时间变化的拓扑距离
    for x, i in enumerate(winner):
      to_update = self.getneighbor(i, N)
      for j in range(N+1):
        e = self.Geteta(t, j)   #表示学习率
        for w in to_update[j]:
          self.W[:, w] = np.add(self.W[:,w], e*(X[x,:] - self.W[:,w]))
  
  def save(self,path):
    """
    :param path: 保存参数文件的路径，npy格式
    """
    np.save(path,self.W)
 
  def getneighbor(self, index, N):
    """
    :param index:获胜神经元的下标
    :param N: 邻域半径
    :return ans: 返回一个集合列表，分别是不同邻域半径内需要更新的神经元坐标
    """
    a, b = self.output
    length = a*b
    def distence(index1, index2):
      i1_a, i1_b = index1 // a, index1 % b   #//:向下取整; %:返回除法的余数;
      i2_a, i2_b = index2 // a, index2 % b
      return np.abs(i1_a - i2_a), np.abs(i1_b - i2_b)   #abs() 函数返回数字的绝对值。
 
    ans = [set() for i in range(N+1)]
    for i in range(length):
      dist_a, dist_b = distence(i, index)
      if dist_a <= N and dist_b <= N: ans[max(dist_a, dist_b)].add(i)
    return ans
 
  def train(self):
    """
    train_Y:训练样本与形状为batch_size*(n*m)
    winner:一个一维向量，batch_size个获胜神经元的下标
    :return:返回值是调整后的W
    """
    count = 0
    while self.iteration > count:
      train_X = self.X[np.random.choice(self.X.shape[0], self.batch_size)]
      normal_W(self.W)
      normal_X(train_X)
      train_Y = train_X.dot(self.W)
      winner = np.argmax(train_Y, axis=1).tolist()
      self.updata_W(train_X, count, winner)
      count += 1
    return self.W
 
  def train_result(self):
    """
    :return winner: 返回最终分类结果
    """
    normal_X(self.X)
    train_Y = self.X.dot(self.W)
    winner = np.argmax(train_Y, axis=1).tolist()
    return winner

  def test_result(self,test,W):
    """
    :param test: 测试集数据
    :param W: 加载的参数
    :return winner: 返回最终分类结果
    """
    X = normal_X(test)
    train_Y = X.dot(W)
    winner = np.argmax(train_Y, axis=1).tolist()
    return winner

def normal_X(X):
  """
  :param X:二维矩阵，N*D，N个D维的数据
  :return: 将X归一化的结果
  """
  N, D = X.shape
  for i in range(N):
    temp = np.sum(np.multiply(X[i], X[i]))
    X[i] /= np.sqrt(temp)
  return X

def normal_W(W):
  """
  :param W:二维矩阵，D*(n*m)，D个n*m维的数据
  :return: 将W归一化的结果
  """
  for i in range(W.shape[1]):
    temp = np.sum(np.multiply(W[:,i], W[:,i]))
    W[:, i] /= np.sqrt(temp)
  return W

def save_dict(transform):
  """
  :param transform: 保存转换矩阵
  """
  jsObj = json.dumps(transform)  
              
  fileObject = open('transform.json', 'w')  
  fileObject.write(jsObj)  
  fileObject.close()  

def train(net,train_loader,format = 'train'):
  """
  :param net: 训练的特征提取网络
  :param train_loader: 训练数据集
  :param format: 选择模型是训练还是测试
  :return som: 返回定义好的特征映射网（初始化）
  """
  final_re = 0
  for epoch in range(10):
      print("epoch: ",epoch)
      torch.cuda.empty_cache()
      acc_sum, n = 0.0, 0
      for X, y in train_loader:

        with torch.no_grad():
          # 若是scrathing方式，则 output = X.view(X.size(0),-1).detach().numpy()
          # output = net(X).numpy()
          # print(X)
          X,y=Variable(X),Variable(y)
          X, y = X.cuda(), y.cuda()
          output, reconstructions, masked=net(X)
          # # output=(net.digit_capsules.forward(X))
          # output = output.detach().numpy().reshape(output.shape[0], -1)
          output=output.detach().cpu().numpy()
          output=output.reshape(output.shape[0],-1)
        # print((output.detach().cpu().numpy()))
        # output = X.view(X.size(0), -1).detach().numpy()
        # output = net(X).detach().numpy()
        if(format=='eval'):
          som = SOM(output, (1, 10), 100, 3000)
          break
        som = SOM(output, (1, 10), 100, 3000)
        som.train()
        res = som.train_result()
        transform = {}
        for i in range(len(res)):
            if res[i] not in transform.keys():
              transform[res[i]]=y[i].item()

        for j in range(len(res)):
            if transform[res[j]] == y[j].item():
              acc_sum+=1
        n += y.shape[0]   
        final_result = acc_sum/n

        if final_result>final_re:
          final_re=final_result
          som.save("w.npy")
          save_dict(transform)
          print("当前最佳训练集正确率为：",final_result)
      if(format == 'eval'):
          break
  return som

def test(net,loader,som):
  """
  :param net: 导入的CNN网络（特征提取用）
  :param loader: 数据集
  :param som: 定义好的自组织特征映射网
  """
  torch.cuda.empty_cache()
  acc_sum,n = 0,0
  W = np.load('w.npy') # 导入权重矩阵
  f2 = open('transform.json', 'r') #导入转换字典
  transform =  json.load(f2)
  for X, y in loader:
    # 若是scrathing方式，则 output = X.view(X.size(0),-1).detach().numpy()
    # output = X.view(X.size(0), -1).detach().numpy()
    # output = net(X).numpy()

    with torch.no_grad():
      X, y = Variable(X), Variable(y)
      X, y = X.cuda(), y.cuda()
      output, reconstructions, masked = net(X)
      output=output.detach().cpu().numpy()
      output = output.reshape(output.shape[0], -1)
      # output = net(X).detach().numpy()
      test = som.test_result(output,W)
    for j in range(len(test)):
        if transform[str(test[j])] == y[j].item():
          acc_sum+=1
    n += y.shape[0]
    print("测试集正确率为：",acc_sum/n)
    torch.cuda.empty_cache()
    break


if __name__ == "__main__":
  # ========================= 导入训练好的网络参数 =========================
  # net = network_for_features()
  dataset = 'mnist'
  config = Config(dataset)

  net=CapsNet(config).cuda()
  net = torch.nn.DataParallel(net)
  net = net.module
  # net = torch.nn.DataParallel(net)
  try:
    net.load_state_dict(torch.load("trained_model/feature_2.pth"))
  except:
    pass

  # ========================= 下载数据集 =========================
  test_dataset = datasets.MNIST(root='./num/',
                                train=False,
                                transform=transforms.ToTensor(),
                                download=True)
  train_dataset = datasets.MNIST(root='./num/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

  # ========================= 装载数据集 =========================
  batch_size =500
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

  test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
  # ========================= SOM模型训练 =========================
  # IMPORTANT!!!
  # format = "eval" 为进行测试集测试情况，目的在于初始化SOM网络
  # format = "train" 为进行训练情况，会更新转换字典和参数矩阵
  # som=train(net,train_loader,format='eval')
  som=train(net,train_loader,format='train')

  # ========================= SOM模型测试 =========================
  test(net,test_loader,som)
  
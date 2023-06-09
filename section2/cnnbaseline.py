# -*- coding: utf-8 -*-
"""CNNbaseline.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16GEt9xOjcXABxpsRh_F5c5pNE6GFyiiR

## 1.モジュールの読み込み、グーグルドライブのマウント
### dataはgoogledriveに格納
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import torch
from torch.utils.data import TensorDataset,DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim
import os

import glob as glob
import zipfile
import shutil
import torchvision

import PIL

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount('/content/drive')

import torchvision.transforms as transforms
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(
        [0.5,0.5,0.5],
        [0.5,0.5,0.5],
    )
]
)

"""driveからカレントディレクトリにtest1.zipとtrain.zipを持ってくる


!unzipを使ってzipファイルを解凍する
"""

train_dataset = torchvision.datasets.CIFAR10(root = './drive/MyDrive/CIFAR10',
                                             train = True,
                                             download=True,
                                             transform=transformer
                                             )

test_dataset = torchvision.datasets.CIFAR10(root = './drive/MyDrive/CIFAR10',
                                            train=False,
                                            download=True,
                                            transform=transformer)

image,label = train_dataset[0]
print("image size:{}".format(image.size()))
print("label: {}".format(label))

train_batch = torch.utils.data.DataLoader(dataset=train_dataset,
                                     batch_size=128,
                                     shuffle=True,
                                     num_workers=2)
test_batch = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=128,
                                         shuffle=False,
                                         num_workers=2)

for im,labels in train_batch:
  print("batch_image_size:{}".format(image.size()))
  print("image size:{}".format(image[0].size()))
  print("batch label size:{}".format(labels.size()))
  break

class BasicBlock(nn.Module):
  expansion = 1
  def __init__(self,in_dim,out_dim,stride=1):
    """
    もしサイズを小さくする場合strideを2にする。
    単純に先にself.size_downを定義してからstrideにifを突っ込めば計算量は増えるけどわざわざforwardでstride=2をいちいち定義しなくてよくなる。
    """
    super(BasicBlock,self).__init__()
    self.conv1 = nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1,stride=stride)
    self.bn1 = nn.BatchNorm2d(out_dim)
    self.relu = nn.ReLU(inplace = True)
    self.conv2 = nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1)
    self.bn2 = nn.BatchNorm2d(out_dim)
    """
    x+f(x)においてf(x)のサイズが小さくなった時に合わせて元のxのサイズを変える。stride=2にすることで単純に画像を実質リサイズしてる。
    poolingでもできると思うからここは検討の余地あり。
    """
    self.down = nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=2)
    self.size_down = (in_dim != out_dim)
    self.stride = stride

  def forward(self,x):
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.size_down == True:
      identity = self.down(identity)
    out += identity
    out = self.relu(out)

    return out

class ResNet20(nn.Module):
  def __init__(self):
    super(ResNet20,self).__init__()

    self.first_Conv = nn.Conv2d(3,16,kernel_size=3,padding=1,stride=1)
    self.first_bn = nn.BatchNorm2d(16)
    self.first_relu = nn.ReLU(inplace=True)

    self.resblock1 = BasicBlock(16,16)
    self.resblock2 = BasicBlock(16,16)
    self.resblock3 = BasicBlock(16,32,stride=2)

    self.resblock4 = BasicBlock(32,32)
    self.resblock5 = BasicBlock(32,32)
    self.resblock6 = BasicBlock(32,64,stride=2)

    self.resblock7 = BasicBlock(64,64)
    self.resblock8 = BasicBlock(64,64)
    self.resblock9 = BasicBlock(64,64)

    self.last_pool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(64,10)

    self.softmax = nn.Softmax(dim=1)

  def forward(self,x):
      x = self.first_relu(self.first_bn(self.first_Conv(x)))
      x = self.resblock1(x)
      x = self.resblock2(x)
      x = self.resblock3(x)
      x = self.resblock4(x)
      x = self.resblock5(x)
      x = self.resblock6(x)
      x = self.resblock7(x)
      x = self.resblock8(x)
      x = self.resblock9(x)

      x = self.last_pool(x)

      x = x.view(x.size(0),-1)
      x = self.fc(x)

      x = self.softmax(x)

      return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = ResNet20().to(device)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0001)

for images,labels in test_batch:
  print(type(images),images.shape)
  print(len(labels))
  print(labels[0])
  break

train_loss_list = []
train_accuracy_list = []
test_loss_list = []
test_accuracy_list = []

epoch = 300
for i in range(epoch):
  print('---------------------')
  print("Epoch:{}/{}".format(i+1,epoch))

  train_loss = 0
  train_accuracy = 0
  test_loss = 0
  test_accuracy = 0


  net.train()
  for images,labels in tqdm(train_batch):
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    y_pred_prob = net(images)

    loss = criterion(y_pred_prob,labels)

    loss.backward()

    optimizer.step()

    train_loss += loss.item()

    y_pred_labels = torch.max(y_pred_prob,1)[1]
    
    train_accuracy += torch.sum(y_pred_labels == labels).item()/len(labels)
  
  epoch_train_loss = train_loss/len(train_batch)
  epoch_train_accuracy = train_accuracy/len(train_batch)

  net.eval()
  with torch.no_grad():
    for images,labels in tqdm(test_batch):
      images = images.to(device)
      labels = labels.to(device)

      y_pred_prob = net(images)

      loss = criterion(y_pred_prob,labels)

      test_loss += loss.item()

      y_pred_labels = torch.max(y_pred_prob,1)[1]

      test_accuracy += torch.sum(y_pred_labels == labels).item()/len(labels)

  epoch_test_loss = test_loss/len(test_batch)
  epoch_test_accuracy = test_accuracy/len(test_batch)

  print("Train_Loss:{},Train_accuracy:{}".format(epoch_train_loss,epoch_train_accuracy))
  print("Test_Loss:{},Test_accuracy:{}".format(epoch_test_loss,epoch_test_accuracy))
  
  train_loss_list.append(epoch_train_loss)
  train_accuracy_list.append(epoch_train_accuracy)
  test_loss_list.append(epoch_test_loss)
  test_accuracy_list.append(epoch_test_accuracy)

model_scripted = torch.jit.script(net)
model_scripted.save('ResNet.pt')

model = torch.jit.load('ResNet.pt',map_location=torch.device('cpu'))

transformer_pred = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5,0.5,0.5],
        [0.5,0.5,0.5],
    )
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = model.to(device)

image_pr = Image.open("./flog_sample.png")
image_pr = transformer_pred(image_pr)
image_pr = image_pr.unsqueeze(0)
image_pr = image_pr.to(device)
output = net(image_pr)

f = nn.Softmax(dim=1)
output = f(output)

label_list = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
score,predicted = torch.max(output,1)

print("score is {}".format(float(score)))
print("predicted is {}".format(label_list[int(predicted)]))
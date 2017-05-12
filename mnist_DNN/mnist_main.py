# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:50:26 2017

@author: danjo
"""

from chainer import Variable, optimizers
import chainer.functions as F
import chainer.links as L

import numpy as np

from mnist_get import mnist
from mnist_model import mnist_model

import matplotlib.pyplot as plt


# mnist取得
mnist = mnist()
x_train, t_train, x_test, t_test = mnist.acquisition()
x_train = Variable(x_train)
t_train = Variable(t_train)
x_test  = Variable(x_test)
t_test  = Variable(t_test)


# modelの定義
model = mnist_model()


# 損失関数など
def forward(data, label, model):
    y = model.predict(data)
#    t.data = t.data.astype(np.int32)
    loss = F.softmax_cross_entropy(y, label)
    acc = F.accuracy(y, label)
    
    return loss, acc


# 最適化の準備
optimizer = optimizers.Adam()
optimizer.setup(model)


# 学習
n_epoch = 5
batch_size = 100
N = len(x_train.data)

l1_W = []
l2_W = []
l3_W = []

train_loss = []
train_acc = []


for i in range(1,n_epoch):
    print("epoch %i" %i)
    
    perm = np.random.permutation(N)
    
    for j in range(0, N, batch_size):
        x_batch = x_train[perm[i:i+batch_size]]
        t_batch = t_train[perm[i:i+batch_size]]
    
        optimizer.zero_grads()
        
        loss, acc = forward(x_batch, t_batch, model)
#        print(loss.data)
        optimizer.update()
    
    
        l1_W.append(model.l1.W)
        l2_W.append(model.l2.W)
        l3_W.append(model.l3.W)
        
        train_loss.append(loss.data)
        train_acc.append(acc.data)
        
# for plot loss function
x = np.linspace(-10,10,len(train_loss),np.float32)
plt.plot(x, train_loss)
plt.plot(x, train_acc)
plt.show()
    

    
#    print("Accuracy : %f" % acc.data)
#    print("Loss : %f" % loss.data)

#print(l1_W,l2_W,l3_W)


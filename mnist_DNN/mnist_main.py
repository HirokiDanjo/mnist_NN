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
import matplotlib.cm as cm


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
def forward(data, label, model,train=True):
    y = model.predict(data, train)
    loss = F.softmax_cross_entropy(y, label)
    acc = F.accuracy(y, label)
    
    return loss, acc


# 最適化の準備
optimizer = optimizers.Adam()
optimizer.setup(model)


# 学習
n_epoch = 1
batch_size = 100
N_train = len(x_train.data)
N_test = len(x_test.data)

l1_W = []
l2_W = []
l3_W = []

train_loss = []
train_acc = []
test_loss = []
test_acc = []


for i in range(1,n_epoch):
    print("epoch %i" %i)
    
    perm = np.random.permutation(N_train)
    
    for j in range(0, N_train, batch_size):
        x_batch = x_train[perm[i:i+batch_size]]
        t_batch = t_train[perm[i:i+batch_size]]
    
        optimizer.zero_grads()
        
        loss, acc = forward(x_batch, t_batch, model)
        
        #これがなかったから勾配を計算する準備ができていなかった！！！！
        #だから値の更新もおこらなかった！！！！！！！
        loss.backward()

        optimizer.update()
    
        l1_W.append(model.l1.W)
        l2_W.append(model.l2.W)
        l3_W.append(model.l3.W)
        
        train_loss.append(loss.data)
        train_acc.append(acc.data)
    
    for i in range(0, N_test, batch_size):
        x_batch = x_test[i:i+batch_size]
        t_batch = t_test[i:i+batch_size]
    
        loss, acc = forward(x_batch, t_batch, train=False)
        
        test_loss.append(loss)
        test_acc.append(acc)
        
"""            
print("-------------------")    
print("l1.W", model.l1.W.data)    
print("l2.W", model.l2.W.data)        
print("l3.W", model.l3.W.data)        
print("-------------------")   
"""

"""
# for plot loss and accuracy
plt.plot(range(len(train_acc), train_acc, label="train_acc")
plt.plot(range(len(test_acc), test_acc, label="test_acc")
plt.legend()
plt.show()
"""
   
#    print("Accuracy : %f" % acc.data)
#    print("Loss : %f" % loss.data)

#print(l1_W,l2_W,l3_W)



#テスト画像について
def test_mnist(image, n, ans, recog):
    size = 28
#==============================================================================
#  表示方法を疑似カラープロットでしようとしたがうまくいかなかった 
#     Z = image.reshape(size,size)
#     print(Z.data.ndim)
#     Z = Z[::-1,:]
#     plt.subplot(10,10,n)
#     plt.xlim(0,27)
#     plt.ylim(0,27)
#     
#==============================================================================
    img = image.data

    plt.subplot(10,10,n)
    plt.axis('off')
    img = image.data.reshape(size,size)

    plt.imshow(img, cmap=cm.gray_r, interpolation='nearest')

    if ans.data != recog:
        plt.title("False",size=8) # 判定が間違っているものには「False」と表記
    else:
        plt.title('%i' % ans.data)

    
plt.figure(figsize=(15,15))

cnt = 0
for i in np.random.permutation(N_test)[:100]:
    
    x = x_test[i].data.astype(np.float32)
    y = model.predict(x.reshape(1,784), train=False)
    cnt += 1
    test_mnist(x_test[i], cnt, t_test[i], np.argmax(y.data))
    
plt.show()
    
    
    
























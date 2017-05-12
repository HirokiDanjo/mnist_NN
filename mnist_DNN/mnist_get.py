# -*- coding: utf-8 -*-
"""
Created on Mon May  8 14:48:27 2017

@author: danjo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import fetch_mldata

class mnist:
    
    def __init__(self):
        print("mnist_get : OK")
        pass
    
    def acquisition(self):
        """mnistを取得"""
                   
        # mnistデータをロード
        mnist = fetch_mldata('MNIST original', data_home=".")
        mnist.data = mnist.data.astype(np.float32)
        mnist.data /= 255
        
        mnist.target = mnist.target.astype(np.int32)
        
        # training用とtest用に分割
        N = 60000
        x_train, t_train = mnist.data[:N,:], mnist.target[:N]
        x_test, t_test   = mnist.data[N:,:], mnist.target[N:]
        
        return x_train, t_train, x_test, t_test
    
    
    def draw_random(self, training=True, num=25):
        """mnistデータをランダムに描画"""
        
        x_train, t_train, x_test, t_test = self.acquisition()
        
        if training:
            data = x_train
            target = t_train
        else:
            data = x_test
            target = t_test

        
        p = np.random.randint(0, len(data), num)
        img = np.array(list(zip(data, target)))[p]
            
        for index, (image, label) in enumerate(img):
            plt.subplot(5,5,index+1)
            plt.axis('off')
            img = image.reshape(28,28)
            plt.imshow(img, cmap=cm.gray_r, interpolation='nearest')
            plt.title('%i' % label)
            
        plt.show()
    














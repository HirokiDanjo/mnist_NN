# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:53:32 2017

@author: danjo
"""
import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np

# model : 2層の全結合層
class mnist_model(chainer.Chain):
    
        def __init__(self):
            super(mnist_model, self).__init__(
                    l1 = L.Linear(784, 500),
                    l2 = L.Linear(500, 100),
                    l3 = L.Linear(100, 10)
            )
            print("mnist_model : OK")
            
        def predict(self, x):
            train = True
            h1 = F.dropout(F.relu(self.l1(x)), train=train)
            h2 = F.dropout(F.relu(self.l2(h1)), train=train)
            
            return self.l3(h2)

import numpy as np
import matplotlib.pyplot as plt
import math

import chainer
import chainer.links as L
import chainer.functions as F

from chainer import optimizers
#from chainer.optimizer_hooks import WeightDecay

from chainer.datasets import TupleDataset
from chainer.iterators import SerialIterator

from chainer.serializers import save_npz


# Encoder & Decoder
class Encoder(chainer.Chain):
    def __init__(self, w):
        super().__init__()
        with self.init_scope():

            self.conv1 = L.Convolution2D(3, 8, ksize=3, pad=1, stride=2)
            self.conv2 = L.Convolution2D(8, 16, ksize=3, pad=1, stride=2)
            self.conv3 = L.Convolution2D(16, 32, ksize=3, pad=1, stride=2)
            self.conv4 = L.Convolution2D(32, 64, ksize=3, pad=1, stride=2)
            self.conv5 = L.Convolution2D(64, 128, ksize=3, pad=1, stride=2)
            self.linear1 = L.Linear(128*2*2, 100)
            self.linear2 = L.Linear(100, 100)
            self.linear3 = L.Linear(100, 128)
            

    def __call__(self, x):
    
        # x -> [batch, 3 (RGB), h=64, w=64]
        h = F.relu(self.conv1(x))   # [batch, 8, 32, 32]
        h = F.relu(self.conv2(h))   # [batch, 16, 16, 16]
        h = F.relu(self.conv3(h))   # [batch, 32, 8, 8]
        h = F.relu(self.conv4(h))   # [batch, 64, 4, 4]
        h = F.relu(self.conv5(h))   # [batch, 128, 2, 2]
        h = F.relu(self.linear1(h)) # [batch, 100]
        h = F.relu(self.linear2(h)) # [batch, 100]
        h = self.linear3(h)         # [batch, 128]

        return h


class Decoder(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.linear_init1 = L.Linear(128, 128)
            self.linear_init2 = L.Linear(128, 128)
            self.linear_init3 = L.Linear(128, 128)
            self.linear_init4 = L.Linear(128, 128)
            
            self.linear_w1 = L.Linear(128, 3 * 128)
            self.linear_w2 = L.Linear(128, 128 * 128)
            self.linear_w3 = L.Linear(128, 128 * 128)
            self.linear_w4 = L.Linear(128, 128 * 128)
            self.linear_w5 = L.Linear(128, 128 * 1)
            self.linear_b1 = L.Linear(128, 128)
            self.linear_b2 = L.Linear(128, 128)
            self.linear_b3 = L.Linear(128, 128)
            self.linear_b4 = L.Linear(128, 128)
            self.linear_b5 = L.Linear(128, 1)

    def __call__(self, x):
        # x -> [batch size, hidden size=128]
        x = F.relu(self.linear_init1(x))
#        x = F.relu(self.linear_init2(x))
#        x = F.relu(self.linear_init3(x))
        x = self.linear_init4(x)

        params = {}
        params['W1'] = self.linear_w1(x)  # [bs, 3 * 128]
        params['b1'] = self.linear_b1(x)  # [bs, 128]
        params['W2'] = self.linear_w2(x)  # [bs, 128 * 128]
        params['b2'] = self.linear_b2(x)  # [bs, 128]
        params['W3'] = self.linear_w3(x)  # [bs, 128 * 128]
        params['b3'] = self.linear_b3(x)  # [bs, 128]
        params['W4'] = self.linear_w4(x)  # [bs, 128 * 128]
        params['b4'] = self.linear_b4(x)  # [bs, 128]
        params['W5'] = self.linear_w5(x)  # [bs, 128 * 1]
        params['b5'] = self.linear_b5(x)  # [bs, 1]
        return params


def Occupancy_networks(params, o, d, num_sample):
    
    #o, d -> [bs, 3, 600]
    xp = chainer.cuda.get_array_module(o)
    #distances = xp.arange(0 + 1, num_sample + 1).astype('float32')  * 2.6 / num_sample  # [0.26, ... 2.6]
    #distances = xp.arange(0, num_sample).astype('float32')  * 1.7322 / (num_sample-1) + (1.30 - 0.8660)
    distances = xp.arange(0, num_sample).astype('float32')  * 1.7322 / (num_sample-1) + (2.00 - 0.8660)
    x = o[:, :, None, :] + d[:, :, None, :] * distances[None, None, :, None]
    # x -> [bs, 3(xyz), num_sample, 600]
    # print(x.shape)
    
    # Parameters
    w1 = params['W1']  # [bs, 3 * 128]
    w1 = F.reshape(w1, (w1.shape[0], 3, 128))  # [bs, 3, 128]
    w2 = params['W2']
    w2 = F.reshape(w2, (w2.shape[0], 128, 128))
    w3 = params['W3']
    w3 = F.reshape(w3, (w3.shape[0], 128, 128))
    w4 = params['W4']
    w4 = F.reshape(w4, (w4.shape[0], 128, 128))
    w5 = params['W5']
    w5 = F.reshape(w5, (w5.shape[0], 128, 1))
    
    b1 = params['b1'][:, None, :] # [bs, 1, 128]
    b2 = params['b2'][:, None, :]
    b3 = params['b3'][:, None, :]
    b4 = params['b4'][:, None, :]
    b5 = params['b5'][:, None, :]
    
    
    ### Forward ###
    bs = x.shape[0]
    num_rays = x.shape[3] # 600
    
    h = x.transpose((0, 2, 3, 1))  # [bs, #s, 600, 3]
    h = h.reshape((bs, -1, 3)) # [bs, #s*600, 3]
    #print(h.shape)
    
    h = F.batch_matmul(h, w1)
    h = h + b1
    h = F.relu(h)
    h = F.batch_matmul(h, w2)
    h = h + b2
    h = F.relu(h)
    h = F.batch_matmul(h, w3)
    h = h + b3
    h = F.relu(h)
    h = F.batch_matmul(h, w4)
    h = h + b4
    h = F.relu(h)
    h = F.batch_matmul(h, w5)
    h = h + b5
    h = F.sigmoid(h)
    
    return h.reshape(bs, num_sample, num_rays)


class DRS(chainer.Chain):
    def __init__(self, w):
        super().__init__()
        with self.init_scope():
            self.encoder = Encoder(w)
            self.decoder = Decoder()

    def __call__(self, A, B, C, num_sample):
        # C -> [batch size, 3, 64, 64]
        v = self.encoder(C)
        # v -> [batch size, hidden size=128]
        param = self.decoder(v)

        occupancy = Occupancy_networks(param, A, B, num_sample) # [batch, num_smaple, 600]
        
        #trans_prob = F.sum(F.softplus(-occupancy), axis=1)
        trans_prob = F.prod(occupancy, axis=1) # -> [batch, 600]
        # l = 1.7322 / (num_sample-1)
        l = 9.0 / (num_sample-1)
        trans_prob = (trans_prob + 1e-8)**l # trans_prob = trans_prob^l
        
        return trans_prob

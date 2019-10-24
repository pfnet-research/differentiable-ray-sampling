import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import math

import chainer
import chainer.links as L
import chainer.functions as F

from chainer import optimizers
from chainer.datasets import TupleDataset
from chainer.iterators import SerialIterator
from chainer.serializers import save_npz

from model import DRS
from data_loader import Data_loader, Get_name_list

import random
import sys
import time

# GPU or CPU
gpu_id = -1
args = sys.argv
gpu_id = int(args[1])
print ("GPU :", gpu_id)

if (gpu_id == 0):
    import cupy as cp
    import chainer.cuda

print("---------- Finished import modules ----------")
# --------------------------------------------------------------------------------

# Pixels: w×h
w = 64
h = 64
h_image = 224
w_image = 224

# Parameters
num_batch_instance = 8 # 8 instances
num_image = 5          # 5 observations
ray_samples = 3000     # ray per instance
ray_per_image = int(ray_samples/num_image) # ray per observation

# Get name of folder
all_instance_list = Get_name_list(mode='Train')
list_length = len(all_instance_list)
print(list_length)
all_valid_instance_list = Get_name_list(mode='Valid')
valid_list_length = len(all_valid_instance_list)
print(valid_list_length)

# initialize DRS networks
drs = DRS(w)

if (gpu_id == 0):
    drs.to_gpu(gpu_id)

index = 0
loss_train_list = []
loss_valid_list = []

optimizer = optimizers.Adam(alpha=0.0001,beta1=0.9,beta2=0.999)
optimizer.setup(drs)

iterations = 30000
num_sample = 40 # (if =1, NaN appears)

# Start trainig
for i in range(iterations):

    train_list = []
    for j in range(num_batch_instance):
        if (index == list_length):
            random.shuffle(all_instance_list) # shuffle
            index = 0
        train_list.append(all_instance_list[index])
        index += 1
    #print(train_list)

#    start = time.time()
    # A: origin, B: direction, C: RGB, D: Mask
    A, B, C, D = Data_loader(train_list, num_image, ray_per_image, h, w, h_image, w_image, gpu_id, 'Train')
    # A,B -> [bs, 3, 600]   (bs = 8*5)
    # C   -> [bs, 3, 64, 64]
    # D   -> [bs, 600]
    #print(A.shape, B.shape, C.shape, D.shape)
#    elapsed_time = time.time() - start
#    print (" elapsed_time:{0}".format(elapsed_time) + "[sec]")
    if (gpu_id==0):
        A = chainer.cuda.to_gpu(A)
        B = chainer.cuda.to_gpu(B)
        C = chainer.cuda.to_gpu(C)
        D = chainer.cuda.to_gpu(D)

    # Forward
    start = time.time()
    output = drs(A, B, C, num_sample) # [bs, 3000]

    # Loss
    loss_train = - F.sum(D*F.log(output + 1e-8) + (1.0-D)*F.log(1.0-output + 1e-8)) / (num_batch_instance*ray_samples)
    #loss_train = - F.sum(D*F.log(output + 1e-8) + (1.0-D)*F.log(1-output + 1e-8)) / (224*224)

    # Grad
    drs.cleargrads()
    loss_train.backward()

    # Update
    optimizer.update()

    #elapsed_time = time.time() - start
    #print ("2 elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # Evaluation using valid datas
    num_valid_instance = 2
    temp_index = randint(0, valid_list_length, num_valid_instance)
    valid_list = [all_valid_instance_list[temp_index[i]] for i in range(num_valid_instance)]
    #print(valid_list)
    A, B, C, D = Data_loader(valid_list, num_image, ray_per_image, h, w, h_image, w_image, gpu_id, 'Valid')
    if (gpu_id==0):
        A = chainer.cuda.to_gpu(A)
        B = chainer.cuda.to_gpu(B)
        C = chainer.cuda.to_gpu(C)
        D = chainer.cuda.to_gpu(D)
    output = drs(A, B, C, num_sample)
    loss_valid = - F.sum(D*F.log(output + 1e-8) + (1.0-D)*F.log(1-output + 1e-8)) / (num_valid_instance*ray_samples)

    # Print
    if (gpu_id==0):
        loss_train.to_cpu()
        loss_valid.to_cpu()
    print("{:5d} iteration, Loss (train, valid): ({:.3f}, {:.3f})".format(i+1, loss_train.data, loss_valid.data))
    loss_train_list.append(loss_train.data)
    loss_valid_list.append(loss_valid.data)

    # Save networks
    if ((i+1)%5000==0):
        print("save npz...")
        save_npz('./save/save' + str(i+1) + '.npz', drs)
        with open('./save/loss_train.txt', mode='w') as f:
            for i in range(len(loss_train_list)):
                f.write(str(loss_train_list[i])+'\n')
        with open('./save/loss_valid.txt', mode='w') as f:
            for i in range(len(loss_valid_list)):
                f.write(str(loss_valid_list[i])+'\n')
        print("finished.")

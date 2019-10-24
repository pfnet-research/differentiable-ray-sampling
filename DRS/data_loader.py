import numpy as np
#import cupy as cp
from numpy.random import *

import scipy.io
from skimage import io, color
from sklearn.preprocessing import normalize
from io import BytesIO
from scipy.misc import imresize

import os
import time

def Get_path(mode):
    path_dataset = '../Dataset/02958343/'
    if (mode=='Train' or mode=='Valid' or mode=='Test'):
        path_dataset += mode + '/'
    else:
        print("Error occurred in Get_path function.")
    return path_dataset

def Get_name_list(mode):
    path = Get_path(mode)
    name_list = os.listdir(path)
    try:
        name_list.remove('.DS_Store')
    except ValueError:
        pass
    
    return name_list


def Load_RGB(path, h, w):
    
    img = io.imread(path)
    img = color.rgba2rgb(img) # RGB [224, 224, 3]
    img = imresize(img, (h, w), interp='bilinear')
    
    return img/255.0


def Load_Mask(path, ray_per_image, random_uv):
    
    img = io.imread(path)

    output = np.array(  [img[random_uv[1,i], random_uv[0,i]] for i in range(ray_per_image) ], dtype='float32' )
    output[output<65535.0-1e-04] = 0.0
    output[output>=65535.0-1e-04] = 1.0
    
    return output


def Load_Camera(path, ray_per_image, random_uv, gpu_id):

    # mat
    mat_data = scipy.io.loadmat(path)
    
    length = ray_per_image
    o = np.empty((3, length), dtype='float32')
    d = np.empty((3, length), dtype='float32')
    
    K = mat_data['K']
    R = mat_data['extrinsic'][0:3,0:3]
    T = np.array([mat_data['extrinsic'][0:3,3]]).transpose(1,0)
    K_inv = np.linalg.inv(K)
    R_inv = np.linalg.inv(R)
    RK_inv = np.dot(R_inv, K_inv)

    origin = -np.dot(R.transpose(1,0), T)
    for index in range(length):
        o[:,index] = origin[:,0]
    
    temp = random_uv + 0.5
    pixel_index = np.ones((3, ray_per_image))
    pixel_index[0:2] = temp
    #pixel_index = np.array([[random_u[i]+0.5 for i in range(length)],[random_v[i]+0.5 for i in range(length)],[1.0 for i in range(length)]])
    d = np.dot(RK_inv, pixel_index)
    d = normalize(d, norm='l2',axis=0) # normalization sklearn https://qiita.com/panda531/items/4ca6f7e078b749cf75e8
    
    return o, d


# (name_list, 5, 600, 64, 64, 224, 224)
def Data_loader(name_list, num_image, ray_per_image, h, w, h_image, w_image, gpu_id, mode):
    
    temp_index = 0
    if (ray_per_image>3000):
        temp_index = int(ray_per_image/(224*224)) - 1
        ray_per_image = 224*224

    length = len(name_list)

    A = np.empty((length*num_image, 3, ray_per_image), dtype='float32') # (8*5, 3, 600)
    B = np.empty((length*num_image, 3, ray_per_image), dtype='float32') # (8*5, 3, 600)
    C = np.empty((length*num_image, 3, h, w), dtype='float32')          # (8*5, 3, 64, 64)
    D = np.empty((length*num_image, ray_per_image), dtype='float32')    # (8*5, 600)
    
    for i in range(length): # loop for each instance
        
        # path
        path_data = Get_path(mode) # '../../Dataset/02958343/'
        id_instance = name_list[i]
        id_image = randint(5)    # choose from {0,1,2,3,4} randomly
        id_mat = np.random.permutation(np.arange(num_image)) # sort {0,1,2,3,4} randomly
        if (ray_per_image==224*224):
            id_image = randint(5)
            id_mat = np.array([temp_index])
    
        # ---------- RGB Image ----------
        path_rgb = path_data + id_instance + '/render_{}.png'.format(id_image)
        RGB_image = Load_RGB(path_rgb, h, w).transpose(2,0,1) # (3, h, w)に変更
        
        for j in range(num_image): # 1 instance, 5 observations
            
            path_depth = path_data + id_instance + '/depth_{}.png'.format(id_mat[j])
            path_mat = path_data + id_instance + '/camera_{}.mat'.format(id_mat[j])
            
            random_uv = randint(0, 224, (2,ray_per_image)) # [2, 600]  (u, v)
            
            if (ray_per_image==224*224):
                random_uv = np.array(  [[int(i%224) for i in range(ray_per_image)],[int(i/224) for i in range(ray_per_image)]]  ) # (0,0) (1,0) ...

            # -------- Mask image --------
            mask_image = Load_Mask(path_depth, ray_per_image, random_uv)  # (600)

            # ---------- Camera ----------
            o, d = Load_Camera(path_mat, ray_per_image, random_uv, gpu_id) # (3, 600)

            A[i*num_image+j] = o
            B[i*num_image+j] = d
            C[i*num_image+j] = RGB_image
            D[i*num_image+j] = mask_image
    
    return A, B, C, D





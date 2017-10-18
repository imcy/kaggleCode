# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
#import image2pkl
#X, Y = image2pkl.load_data(one_hot=True, resize_pics=(224,224))
'''using .pkl file directly'''
import pickle
import numpy as np
import pandas as pd

file = open("/home/robot/cy/species/dataset.pkl",'rb')
data, labels = pickle.load(file)
# Building 'AlexNet'
network = input_data(shape=[None, 224, 224, 3]) # 图片尺寸224x224
network = conv_2d(network, 96, 11, strides=4, activation='relu')  # 卷积层1，卷积核大小11x11,步长4,96个卷积核
network = max_pool_2d(network, 3, strides=2)   # max_pool层1,3x3,步长2
network = local_response_normalization(network)  # LRN
network = conv_2d(network, 256, 5, activation='relu')  # 卷积层2，5x5，256个卷积核
network = max_pool_2d(network, 3, strides=2)  # max_pool层2
network = local_response_normalization(network)  # LRN
network = conv_2d(network, 384, 3, activation='relu')  # 3x3卷积层3
network = conv_2d(network, 384, 3, activation='relu')  # 3x3卷积层4
network = conv_2d(network, 256, 3, activation='relu')  # 3x3卷积层5
network = max_pool_2d(network, 3, strides=2)  # max_pool层3
network = local_response_normalization(network)  # LRN
network = fully_connected(network, 4096, activation='tanh')  # 全连接层1
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')  # 全连接层2
network = dropout(network, 0.5)
network = fully_connected(network, 1, activation='softmax')  # 全连接层3
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='species_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)


model.fit(data, labels, n_epoch=500, validation_set=0.2, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_species')
model.save("./model/model.tfl")
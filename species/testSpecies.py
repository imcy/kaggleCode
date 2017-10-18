from PIL import Image
import numpy as np
import os
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
import pandas as pd

# Building 'VGG Network'
network = input_data(shape=[None, 224, 224, 3])

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')

# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                max_checkpoints=1, tensorboard_verbose=0)
model.load("./model/modelSpecies.tfl") #提取图片特征
directory = "./test/"
print(directory)
pd_data=[]
for i in range(1,1532):
    filename=str(i)+'.jpg'
    print(filename)
    img = Image.open(directory + filename)
    resize_mode = Image.ANTIALIAS
    img = img.resize((224, 224), resize_mode)
    img = np.array([np.array(img)])
    y = model.predict_label(img)
    print(y[0][0])
    pd_data.append(y[0][0])

pd_data=pd.DataFrame(pd_data)
pd_data.to_csv('./predict.csv')


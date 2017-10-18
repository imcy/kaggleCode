# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 19:07:10 2017

@author: Administrator
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import Convolution2D, MaxPooling2D  
from keras.utils import np_utils
import os
import pandas as pd
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from keras import backend as K
import tensorflow as tf

# 全局变量  
batch_size = 100  
nb_classes = 10  
epochs = 20
# input image dimensions  
img_rows, img_cols = 28, 28  
# number of convolutional filters to use  
nb_filters = 32  
# size of pooling area for max pooling  
pool_size = (2, 2)  
# convolution kernel size  
kernel_size = (3, 3)  

inputfile='F:/data/kaggle/mnist/train.csv'
inputfile2= 'F:/data/kaggle/mnist/test.csv'
outputfile= 'F:/data/kaggle/mnist/test_label.csv'


pwd = os.getcwd()
os.chdir(os.path.dirname(inputfile)) 
train= pd.read_csv(os.path.basename(inputfile)) #reading train data from inputfile
os.chdir(pwd)

pwd = os.getcwd()
os.chdir(os.path.dirname(inputfile)) 
test= pd.read_csv(os.path.basename(inputfile2)) #reading train data from inputfile
os.chdir(pwd)

x_train=train.iloc[:,1:785] #Extract the features
y_train=train['label']
y_train = np_utils.to_categorical(y_train, 10)

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True) #导入数据
x_test=mnist.test.images
y_test=mnist.test.labels
# 根据不同的backend定下不同的格式  
if K.image_dim_ordering() == 'th': 
    x_train=np.array(x_train)
    test=np.array(test)
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)  
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)  
    input_shape = (1, img_rows, img_cols)  
    test = test.reshape(test.shape[0], 1, img_rows, img_cols)  
else:  
    x_train=np.array(x_train)
    test=np.array(test)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)  
    X_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) 
    test = test.reshape(test.shape[0], img_rows, img_cols, 1)  
    input_shape = (img_rows, img_cols, 1)  
  
x_train = x_train.astype('float32')  
x_test = X_test.astype('float32')  
test = test.astype('float32')  
x_train /= 255  
X_test /= 255
test/=255  
print('X_train shape:', x_train.shape)  
print(x_train.shape[0], 'train samples')  
print(x_test.shape[0], 'test samples')  
print(test.shape[0], 'testOuput samples')  

model=Sequential()#model initial
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),  
                        padding='same',  
                        input_shape=input_shape)) # 卷积层1  
model.add(Activation('relu')) #激活层  
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]))) #卷积层2  
model.add(Activation('relu')) #激活层  
model.add(MaxPooling2D(pool_size=pool_size)) #池化层  
model.add(Dropout(0.25)) #神经元随机失活  
model.add(Flatten()) #拉成一维数据  
model.add(Dense(128)) #全连接层1  
model.add(Activation('relu')) #激活层  
model.add(Dropout(0.5)) #随机失活  
model.add(Dense(nb_classes)) #全连接层2  
model.add(Activation('softmax')) #Softmax评分  
  
#编译模型  
model.compile(loss='categorical_crossentropy',  
              optimizer='adadelta',  
              metrics=['accuracy'])  
#训练模型  

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1)  
model.predict(x_test)
#评估模型  
score = model.evaluate(x_test, y_test, verbose=0)  
print('Test score:', score[0])  
print('Test accuracy:', score[1])  

y_test=model.predict(test)

sess=tf.InteractiveSession()
y_test=sess.run(tf.arg_max(y_test,1))
y_test=pd.DataFrame(y_test)
y_test.to_csv(outputfile)


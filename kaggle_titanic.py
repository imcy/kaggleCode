# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:29:23 2017

@author: Administrator
"""

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from keras.optimizers import SGD
#import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

inputfile='F:/data/kaggle/ta/train .csv'
inputfile2= 'F:/data/kaggle/ta/test .csv'
outputfile='F:/data/kaggle/ta/test_label.csv'
train=pd.read_csv(inputfile)
test=pd.read_csv(inputfile2)

selected_features=['Pclass','Sex','Age','Embarked','SibSp','Parch','Fare']

x_train=train[selected_features]
y_train=train['Survived']

x_test=test[selected_features]

#补充Embarked缺失值,使用频率最高的特征填充
x_train['Embarked'].fillna('S',inplace=True)
x_test['Embarked'].fillna('S',inplace=True)

#填充Age特征，用求平均值替代
x_train['Age'].fillna(x_train['Age'].mean(),inplace=True)
x_test['Age'].fillna(x_test['Age'].mean(),inplace=True)
x_test['Fare'].fillna(x_test['Fare'].mean(),inplace=True)
#X_train,X_test,Y_train,Y_test=train_test_split(x_train,y_train,test_size=0.25,random_state=33)  
#特征向量化
dict_vec=DictVectorizer(sparse=False)
x_train=dict_vec.fit_transform(x_train.to_dict(orient='record'))
x_test=dict_vec.fit_transform(x_test.to_dict(orient='record'))
#print(dict_vec.feature_names_)
#x_train=(x_train - x_train.min())/(x_train.max() - x_train.min())#minimum - maximum normalization

svc =SVC(kernel='linear')#,C=10,gamma=10
svc.fit(x_train,y_train)
predit_y=svc.predict(x_test)
predit_y=pd.DataFrame(predit_y)
predit_y.to_csv(outputfile)
#print(cross_val_score(clf_sigmoid, x_train, y_train, cv=5).mean())
#print('Accuracy of SVM Classifier:',svc.score(X_test,Y_test))
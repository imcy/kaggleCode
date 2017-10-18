# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:48:03 2017

@author: Administrator
"""

import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer #导入文本特征抽取器
from sklearn.naive_bayes import MultinomialNB #导入朴素贝叶斯模型
from sklearn.pipeline import Pipeline #导入pipeline便于搭建模型
from sklearn.grid_search import GridSearchCV #导入网格搜索，用于超参数搜索

inputfile='labeledTrainData.tsv'
inputfile2= 'testData.tsv'

train=pd.read_csv(inputfile,delimiter='\t')
test=pd.read_csv(inputfile2,delimiter='\t')

#原始评论三项数据预处理任务
def review_to_text(review,remove_stopwords):
    #任务1：去掉html标记
    raw_text=BeautifulSoup(review,'html').get_text()
    #任务2：去掉非字母字符
    letters=re.sub('[^a-zA-Z]',' ',raw_text)
    words=letters.lower().split() #转小写并分割
    #任务3：去掉停用词
    if remove_stopwords:
        stop_words=set(stopwords.words('english'))
        words=[w for w in words if w not in stop_words]
    return words

#对原始训练和测试数据集进行三项处理
x_train=[]
for review in train['review']:
    x_train.append(' '.join(review_to_text(review,True)))
x_test=[]
for review in test['review']:
    x_test.append(' '.join(review_to_text(review,True)))
y_train=train['sentiment']

#countVectorizer
pip_count=Pipeline([('count_vec',CountVectorizer(analyzer='word')),('mnb',MultinomialNB())])
#tfidf
pip_tfidf=Pipeline([('tfidf_vec',TfidfVectorizer(analyzer='word')),('mnb',MultinomialNB())])

params_count={'count_vec__binary':[True,False],'count_vec__ngram_range':
    [(1,1),(1,2)],'mnb__alpha':[0.1,1.0,10.0]}
params_tfidf={'tfidf_vec__binary':[True,False],'tfidf_vec__ngram_range':
    [(1,1),(1,2)],'mnb__alpha':[0.1,1.0,10.0]}
    
gs_count=GridSearchCV(pip_count,params_count,cv=4,n_jobs=-1,verbose=1)
gs_count.fit(x_train,y_train)
print (gs_count.best_score_)
print (gs_count.best_params_)

gs_tfidf=GridSearchCV(pip_tfidf,params_tfidf,cv=4,n_jobs=-1,verbose=1)
gs_tfidf.fit(x_train,y_train)
print(gs_tfidf.best_score_)
print(gs_tfidf.best_params_)
count_y_predict=gs_count.predict(x_test) #最佳参数组合预测
tfidf_y_predict=gs_tfidf.predict(x_test)

submission_count=pd.DataFrame({'id':test['id'],'sentiment':count_y_predict})
submission_tfidf=pd.DataFrame({'id':test['id'],'sentiment':tfidf_y_predict})

submission_count.to_csv('submission_count.csv',index=False)
submission_tfidf.to_csv('submission_tfidf.csv',index=False)

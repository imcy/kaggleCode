from gensim.models import Word2Vec
import numpy as np
import  pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

inputfile='labeledTrainData.tsv'
inputfile2= 'testData.tsv'
train=pd.read_csv(inputfile,delimiter='\t')
test=pd.read_csv(inputfile2,delimiter='\t')

model=Word2Vec.load('300features_20minwords_10context')

set(model.index2Word)
'''
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

y_train=train['sentiment']

#定义一个函数使用词向量产生文本特征向量
def makeFeatureVec(words,model,num_features):
    featureVec=np.zeros((num_features,),dtype="float32")
    nwords=0.
    index2word_set=set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords=nwords+1.
            featureVec=np.add(featureVec,model[word])
    featureVec=np.divide(featureVec,nwords)
    return featureVec
#每一条影评转为基于词向量的特征向量（平均词向量）
def getAvgFeatureVecs(reviews,model,num_features):
    counter=0
    reviewFeatureVecs=np.zeros((len(reviews),num_features),dtype="float32")

    for review in reviews:
        reviewFeatureVecs[counter]=makeFeatureVec(review,model,num_features)
        counter+=1
    return reviewFeatureVecs
clean_train_reviews=[]

for review in train["review"]:
    clean_train_reviews.append(review_to_text(review,remove_stopwords=True))

num_features=300

trainDataVecs=getAvgFeatureVecs(clean_train_reviews,model,num_features)

clean_test_reviews=[]
for review in test["review"]:
    clean_test_reviews.append(review_to_text(review,remove_stopwords=True))

testDataVecs=getAvgFeatureVecs(clean_test_reviews,model,num_features)
from  sklearn.ensemble import GradientBoostingClassifier #导入模型进行情感分析
from sklearn.grid_search import GridSearchCV #网格搜索

gbc=GradientBoostingClassifier()
params_gbc={'n_estimators':[10,100,500],'learning_rate':[0.01,0.1,1.0],'max_depth':[2,3,4]}
gs=GridSearchCV(gbc,params_gbc,cv=4,n_jobs=-1,verbose=1)
gs.fit(trainDataVecs,y_train)

print(gs.best_score_)
print(gs.best_params_)
'''
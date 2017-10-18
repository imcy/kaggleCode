import nltk.data
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

unlabeled_train=pd.read_csv('unlabeledTrainData.tsv',delimiter='\t',quoting=3)

tokenizer=nltk.data.load('tokenizers/punkt/english.pickle') #使用tokenizer对影评进行分割
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
#逐条分句
def review_to_sentences(review,tokenizer):
    raw_sentences=tokenizer.tokenize(review.strip())
    sentences=[]
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(review_to_text(raw_sentence,False))
    return  sentences
corpora=[]

for review in unlabeled_train['review']:
    corpora+=review_to_sentences(review,tokenizer)
num_features=300
min_word_count=20
num_workers=4
context=10
downsampling=1e-3

from gensim.models import word2vec #导入词向量模型
model=word2vec.Word2Vec(corpora,
                        workers=num_workers,
                        size=num_features,
                        min_count=min_word_count,
                        window=context,
                        sample=downsampling) #传入参数
model.init_sims(replace=True)
model_name="300features_20minwords_10context"
model.save(model_name) #模型保存



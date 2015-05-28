# -*- coding: utf-8 -*-
# coding=utf-8
import csv, nltk, numpy as np, re, const_values as const
import vectorizer_estimator as vec_est
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, auc
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
import pylab as pl
import pickle
import customed_vectorizer as cstv
from bs4 import BeautifulSoup
import sys


reload(sys)
sys.setdefaultencoding("utf-8")
# 加载数据
def load_data():
    print('start loading data...')

    # 格式：kaggle的标记数据，电影品论
    inpTweets = csv.reader(open('data/labeledTrainData.csv', 'rt'),delimiter='\t')
    X = []
    Y = []
    for row in inpTweets:
        sentiment = (1 if row[1] == '1' else 0)
        tweet = row[2]
        Y.append(sentiment)
        X.append(tweet)
    # end loop

    movie_review_test = csv.reader(open('data/kaggle_testData.csv', 'rt'),delimiter='\t')
    unlabled_review=[]
    review_id=[]
    for row in movie_review_test:
        review = row[1]
        id_rv=row[0]
        unlabled_review.append(review)
        review_id.append(id_rv)
    # end loop
    pickle.dump(review_id, open("./review_id.p", "wb"))
    return X, Y, unlabled_review

# 预处理
def preprocessor(tweet):

    emo_repl_order = const.emo_repl_order
    emo_repl = const.emo_repl
    re_repl = const.re_repl

    tweet = BeautifulSoup(tweet).get_text()
    tweet = tweet.lower()
    for k in emo_repl_order:
        tweet = tweet.replace(k, emo_repl[k])
    tweet=tweet.replace('\'s ','').replace("-", " ").replace("_", " ").replace('"','').replace(".",'').\
        replace(',','').replace(';','').strip()
    for r, repl in re_repl.items():
        tweet = re.sub(r, repl, tweet)
    return tweet


# 使用Pipeline类，将向量化处理器和分类器结合到一起，其中TfidfVectorizer作用是将原始推文文本转换为Tf-IDF特征值，以便和标签(label)结合后训练分类器
# 返回的Pipeline实例可以用于fit()和predict()，类似于一个分类器作用
def create_ngram_model(params=None):

    print('start create_ngram_model...')
    tfidf_ngrams = TfidfVectorizer(preprocessor=preprocessor, ngram_range=(3, 3), analyzer='word', binary=False,max_features=5000)
    clf = MultinomialNB()
    pipeline = Pipeline([('vect', tfidf_ngrams), ('clf', clf)])
    return pipeline


# 利用Scikit-learn的FeatureUnion类，把TfidfVectorizer和语言特征（LinguisticVectorizer）结合起来，并行处理
def create_union_model(params=None):

    print('start create_union_model...')
    tfidf_ngrams = cstv.StemmedTfidfVectorizer(preprocessor=preprocessor, ngram_range=(2, 2), analyzer='word', binary=False)

    ling_status = vec_est.LinguisticVectorizer()
    all_features = FeatureUnion([('ling', ling_status), ('tfidf', tfidf_ngrams)])

    clf = MultinomialNB()
    pipeline = Pipeline([("all", all_features), ('clf', clf)])

    if params:
        pipeline.set_params(**params)

    return pipeline


#不使用KFold，因为他会把数据切分成连续的几折，相反，使用ShuffleSplit，将数据随机打散，但不能保证相同的数据样本不会出现在多个数据折中
#训练模型，把创建分类器的函数作为参数传入
def train_model(clc_factory, X, Y,testdata):
    print('start train_model...')
    #设置随机状态，来得到确定性的行为
    # just for kaggle
    #cv = ShuffleSplit(n=len(X), n_iter=1, test_size=0.0, indices=True, random_state=0)

    # accuracy
    scores = []
    # AUC
    pr_scores = []
    # F1 score
    f1=[]

    #for train, test in cv:

        # just for kaggle, use all data

    X_test=testdata

    X_train, Y_train = X, Y

    clf = clc_factory()
    clf.fit(X_train, Y_train)

    predict_data=clf.predict(X_test)
    pickle.dump(predict_data, open("./kaggle_predict_label.p", "wb"))


X, Y, test_review = load_data()
Y = np.array(Y)
X = np.array(X)
# train_model(create_ngram_model, X, Y,test_review)
train_model(create_union_model, X, Y, test_review)

print('OK，执行完了')
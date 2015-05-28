# -*- coding: utf-8 -*-
# coding=utf-8
import csv, nltk, numpy as np, re, const_values as const
import pickle
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


X, Y, test_review = load_data()
# Y = np.array(Y)
# X = np.array(X)
# train_model(create_ngram_model, X, Y,test_review)


print('OK，执行完了')
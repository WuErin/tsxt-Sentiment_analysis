__author__ = 'NLP-PC'
from simple_classifier import load_data
import pickle
import numpy as np
import pylab as pl
from parameters import parameters
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import time

st = time.time()

X, Y = load_data('polarity_classification')
X, Y = np.array(X), np.array(Y)

from sklearn.cross_validation import ShuffleSplit

cv = ShuffleSplit(n=len(X), n_iter=1, test_size=0.0, random_state=0)

from Utils import preprocessor
from customed_vectorizer import StemmedTfidfVectorizer

vectorizer = StemmedTfidfVectorizer(preprocessor=preprocessor, ngram_range=parameters['ngram_range'], analyzer='word',
                                    min_df=parameters['min_df'], max_df=parameters['max_df'],
                                    binary=parameters['TF_binary'], norm=parameters['norm'],sublinear_tf=parameters['sublinear_tf'])
from sklearn.naive_bayes import MultinomialNB

for train, test in cv:
    X_train, Y_train = X[train], Y[train]

    from Utils import load_test_data
    X_test, Y_test = load_test_data()

    X_test, Y_test = np.array(X_test), np.array(Y_test)

    if parameters['combine_feature']==True:
        from vectorizer_estimator import StatisticVectorizer
        from sklearn.pipeline import FeatureUnion
        statistic_vec=StatisticVectorizer()
        combined_features =FeatureUnion([('ngrams',vectorizer),('statistic_vec',statistic_vec)])
    else:
        combined_features=vectorizer

    trian_vec = combined_features.fit_transform(X_train)
    pickle.dump(combined_features.get_feature_names(), open('./debug/feature_names.p', 'wb'))
    test_vec = combined_features.transform(X_test)  # use transform for test data, instead of fit_transform

    # clf = pickle.load(open("./acc_tmp/clf_all_data_noclustering.p", "rb"))

    if parameters['classifier']=='svm':
        from sklearn import svm
        clf = svm.SVC()
        clf.fit(trian_vec.toarray(), Y_train)
        pickle.dump(trian_vec, open("./debug/trian_vec.p", "wb"))
        pickle.dump(clf, open("./acc_tmp/clf.p", "wb"))
        true_labels=Y_test
        predict_labels=np.array(clf.predict(test_vec.toarray()))
        precision,recall,fbeta_score,support=precision_recall_fscore_support(true_labels, predict_labels, average='binary')
        print('精确度(Precision):%.3f\n召回率：%.3f\nF值: %.3f'%(precision,recall,fbeta_score))
    else:
        clf = MultinomialNB()
        clf.fit(trian_vec, Y_train)
        pickle.dump(trian_vec, open("./debug/trian_vec.p", "wb"))
        pickle.dump(clf, open("./acc_tmp/clf.p", "wb"))
        print(trian_vec.shape, len(X_test))
        print(test_vec.shape, len(Y_test))

        test_score = clf.score(test_vec, Y_test)
        scores = []
        scores.append(test_score)
        proba = clf.predict_proba(test_vec)

        precision, recall, pr_thresholds = precision_recall_curve(Y_test, proba[:, 1])
        # AUC
        aera_uc = auc(recall, precision)
        pr_scores = []
        pr_scores.append(aera_uc)

        # F1_score
        f1 = []
        f1.append(f1_score(Y_test, clf.predict(test_vec), average='macro'))

        summary = (np.mean(scores), np.mean(pr_scores), np.mean(f1))
        # Area Under Curve （曲线下面的面积）
        print('正确率(Accuracy)：%.3f\nP/R AUC值：%.3f\nF值(Macro-F score)：%.3f' % (summary))
        precision_binary,recall_binary,fbeta_score_binary,support_binary=precision_recall_fscore_support(Y_test, clf.predict(test_vec), average='binary')
        print('精确度(Precision):%.3f\n召回率：%.3f\nF值: %.3f'%(precision_binary,recall_binary,fbeta_score_binary))

        # 画图
        pl.clf()
        pl.plot(recall, precision, label='Precision-Recall curve')
        pl.xlabel('Recall')
        pl.ylabel('Precision')
        pl.ylim([0.0, 1.05])
        pl.xlim([0.0, 1.0])
        pl.title('Precision-Recall Curve (AUC=%0.3f)' % aera_uc)
        pl.legend(loc="lower left")
        pl.show()

    print(sorted(list(parameters.items())))
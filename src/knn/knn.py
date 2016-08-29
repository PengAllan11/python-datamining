# encoding: utf-8
"""
@author: peng_an
@time: 2016/8/28 20:15
"""

import numpy as np
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split


''''' read data '''
def read_data(url):
    data = []
    labels = []
    with open(url) as ifile:
        for line in ifile:
            tokens = line.strip().split(',')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
    return np.array(data),labels


'''train the model and test it'''
def knn_classification(data,labels,test_rate=0.2):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_rate)
    clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
    clf.fit(x_train, y_train)
    answer = clf.predict(data)
    print(labels)
    print(answer.tolist())

    print(classification_report(labels, answer))


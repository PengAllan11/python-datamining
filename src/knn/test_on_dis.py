# encoding: utf-8
"""
@author: peng_an
@time: 2016/9/3 14:11
"""
from numpy import hstack, vstack, array, median, nan
from numpy.random import choice
from sklearn.datasets import load_iris
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn import metrics


df = pd.read_csv('data\\iris.txt')
print df.columns

x = df.drop('Species', axis=1)
y = df.Species

# enc = OneHotEncoder()
# enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
# iris = load_iris()
# print iris.target
# print enc.transform([[1, 1, 3]]).toarray()
v = DictVectorizer()
# print x.to_dict(orient='records')[1]
# test = v.fit_transform(x.to_dict(orient='records'))
# print test[0]
# print test[146].toarray()
# print v.inverse_transform(test[146])
x = v.fit_transform(x.to_dict(orient='records')).toarray()
#
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# train the model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)

predict = model.predict(x_test)

precision = metrics.precision_score(y_test, predict, average='weighted')
recall = metrics.recall_score(y_test, predict, average='weighted')
print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
accuracy = metrics.accuracy_score(y_test, predict)
print('accuracy: %.2f%%' % (100 * accuracy))

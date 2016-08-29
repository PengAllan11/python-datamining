# encoding: utf-8
"""
@author: peng_an
@time: 2016/8/29 19:37
"""
import numpy as np
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np

clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
clf.fit(x_train, y_train)

test=[]
test.append([1,2])
test.append([2,3])
test.append([3,4])
x = np.array(test)
y=np.zeros(5)

labels=['a','b','c','c']

print(len(set(labels)))

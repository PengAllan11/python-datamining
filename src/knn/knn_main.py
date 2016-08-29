# encoding: utf-8
"""
@author: peng_an
@time: 2016/8/28 20:15
"""

import knn as knn

data,labels=knn.read_data("data/body.txt")
knn.knn_classification(data,labels,0.3)

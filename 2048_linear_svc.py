from __future__ import absolute_import, division, print_function, unicode_literals

import math

import pandas as pd
from numpy.random import RandomState

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

from sklearn import svm


def adapt(number):
    if number == 0:
        return 0
    else:
        return math.log2(number) / 16.0


df = pd.read_csv("moves.csv")
df.columns = ['11', '12', '13', '14', '21', '22', '23', '24', '31', '32', '33', '34', '41', '42', '43', '44', 'move']
df = df.drop_duplicates()
print(df.shape)
rng = RandomState()

train = df.sample(frac=0.8, random_state=rng)
test = df.loc[~df.index.isin(train.index)]

train_X = train[['11', '12', '13', '14', '21', '22', '23', '24', '31', '32', '33', '34', '41', '42', '43', '44']]
train_X = train_X.apply(lambda x: x.apply(lambda y: adapt(y)), axis=1)
train_y = train['move']
test_X = test[['11', '12', '13', '14', '21', '22', '23', '24', '31', '32', '33', '34', '41', '42', '43', '44']]
test_X = test_X.apply(lambda x: x.apply(lambda y: adapt(y)), axis=1)
test_y = test['move']

class_names = ['Up', 'Right', 'Down', 'Left']

clf = svm.SVC(kernel='linear', C=1.0)
print("training model")
clf.fit(train_X, train_y)
print("getting accuracy")
test_acc = clf.score(test_X, test_y)

print('\nTest accuracy:', test_acc)
#print('\nTest loss:', test_loss)



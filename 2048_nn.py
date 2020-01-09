from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import math
from numpy.random import RandomState
from keras.models import Sequential
from keras.layers import Dense

def adapt(number):
    if number == 0:
        return 0
    else:
        return math.log2(number) / 16.0


df = pd.read_csv("moves.csv")
df.columns = ['11', '12', '13', '14', '21', '22', '23', '24', '31', '32', '33', '34', '41', '42', '43', '44', 'move']
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
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(test_images.shape)

model = Sequential()
model.add(Dense(32, input_dim=16, activation='tanh'))
model.add(Dense(24, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_X, train_y, epochs=100)

test_loss, test_acc = model.evaluate(test_X,  test_y, verbose=2)

print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)



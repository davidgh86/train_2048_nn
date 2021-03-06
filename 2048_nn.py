from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
from numpy.random import RandomState
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv("moves.csv")
df.columns = ['11', '12', '13', '14', '21', '22', '23', '24', '31', '32', '33', '34', '41', '42', '43', '44', 'move']
df = df.drop_duplicates()
print(df.shape)
rng = RandomState()

train = df.sample(frac=0.8, random_state=rng)
test = df.loc[~df.index.isin(train.index)]

train_X = train[['11', '12', '13', '14', '21', '22', '23', '24', '31', '32', '33', '34', '41', '42', '43', '44']]
train_y = train['move']
test_X = test[['11', '12', '13', '14', '21', '22', '23', '24', '31', '32', '33', '34', '41', '42', '43', '44']]
test_y = test['move']

class_names = ['Up', 'Right', 'Down', 'Left']

model = Sequential()
model.add(Dense(512, input_dim=16, activation='sigmoid'))
# model.add(Dense(256, activation='relu'))
model.add(Dense(1024, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_X, train_y, epochs=1000, verbose=2)

test_loss, test_acc = model.evaluate(test_X,  test_y, verbose=2)

print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)

predictions = model.predict(test_X)
print(predictions)

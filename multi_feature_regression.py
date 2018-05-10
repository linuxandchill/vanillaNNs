from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

from keras.models import Sequential
from keras.layers.core import Dense

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', 
        input_shape=(train_data.shape[1], )))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='rmsprop',
            loss='mse',
            metrics=['mae'])
    return model

import numpy as np
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

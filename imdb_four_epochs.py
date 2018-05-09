from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

#encoding integer sequences into binary matrix
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    #create all zero matrix
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#vectorize labels
y_train = train_labels.astype('float32')
y_test = test_labels.astype('float32')

#INPUT -> RELU 16 -> RELU 16 -> SIGMOID 1 -> PROBABILITY
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop

model = Sequential()
model.add(Dense(16, input_shape=(10000,)))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
        x_train,
        y_train,
        epochs=3,
        batch_size = 512
        )

results = model.evaluate(x_test, y_test)

print(results)



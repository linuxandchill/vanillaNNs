from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for index, sequence in enumerate(sequences):
        results[index, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data) 

vect_train_labels = to_categorical(train_labels)
vect_test_labels = to_categorical(test_labels)

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop

model = Sequential()

#NEED SUFFICIENT INTERMEDIATE UNITS IN LAYER 
# FOR 46 CLASS CLASSIFICATION PROBLEM
model.add(Dense(64, input_shape=(10000,)))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
#N units for N classes
#softmax for probability distribution
model.add(Dense(46))
model.add(Activation('softmax'))

model.compile(optimizer=RMSprop(), 
        loss='categorical_crossentropy',
        metrics=['accuracy'])

#create validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
#labels validation set
y_val = vect_train_labels[:1000]
partial_y_train = vect_train_labels[1000:]

model.fit(partial_x_train,
        partial_y_train,
        epochs=9,
        batch_size = 512,
        validation_data=(x_val, y_val))
results = model.evaluate(x_test, vect_test_labels)

predictions = model.predict(x_test)

print(results)
#print(predictions)
#returns index w maximum value
print(np.argmax(predictions[0])

from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

#decode review from ints to english
word_index = imdb.get_word_index()
reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
decoded_review = ''.join(
        [reverse_word_index.get(i -3, '?') for i in train_data[0]])

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

#validate on training data while training
#to see how model performs on data it's never seen
#####create x validation set
x_validation = x_train[:10000]
partial_x_train = x_train[10000:]

####create label validation set
y_validation = y_train[:10000]
partial_y_train = y_train[10000:]

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
        partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size = 512,
        validation_data = (x_validation, y_validation)
        )

model.save('imdb_val.h5')

import matplotlib.pyplot as plt
history_dict = history.history

loss = history_dict['loss']
val_loss = history_dict['val_loss']

acc = history_dict['acc']
val_acc = history_dict['val_acc']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'bo', label="Training loss")
plt.plot(epochs, val_loss, 'b', label="Validation loss")
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf #clear
plt.plot(epochs, acc, 'bo', label="Training accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation accuracy")
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()





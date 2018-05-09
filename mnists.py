from keras.datasets import mnist
from keras.models import Sequential
from keras import layers
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = Sequential()

model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', 
        loss = 'categorical_crossentropy',
        metrics=['accuracy'])

#reshape images
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

#encode labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#train model
history = model.fit(train_images, train_labels, epochs=5, batch_size=32)
train_loss, train_acc = history.history['loss'], history.history['acc']
print('Train Acc: ', train_acc)
print('Train Loss: ', train_loss)

#run model on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Acc: ', test_acc)
print('Test Loss: ', test_loss)

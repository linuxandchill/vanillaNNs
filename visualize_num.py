from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#train_tuple = (train_images, train_labels) == <class 'tuple'>

num = train_images[2]
import matplotlib.pyplot as plt
plt.imshow(num, cmap=plt.cm.binary)
plt.show()

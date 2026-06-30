# Dataset --> Image processing --> Train Test Data --> Neural Network --> prediction

# importing the dependencies

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import tensorflow as tf


tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix

# loading the MNIST data from keras.dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# printing 10th image
print(x_train[10])
print(x_train[10].shape)

# displaying the image

plt.imshow(x_train[200])
plt.show()
# print the corresponding label
print(y_train[200])

# unique values in Y_train
print(np.unique(y_train))

# unique values in Y_test
print(np.unique(y_test))

# scaling the values

x_train = x_train/255
x_test = x_test/255

# printing the 10th image

print(x_train[10])

# setting up the layers of the Neural  Network

model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(50, activation='relu'),
                          keras.layers.Dense(50, activation='relu'),
                          keras.layers.Dense(10, activation='sigmoid')
])
# compiling the Neural Network

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
# training the Neural Network

model.fit(x_train, y_train, epochs=10)

loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print(x_test.shape)

# first data point in X_test
plt.imshow(x_test[0])
plt.show()

print(y_test[0])

Y_pred = model.predict(x_test)

print(Y_pred.shape)

print(Y_pred[0])

# converting the prediction probabilities to class label

label_for_first_test_image = np.argmax(Y_pred[0])
print(label_for_first_test_image)

# converting the prediction probabilities to class label for all test data points
Y_pred_labels = [np.argmax(i) for i in Y_pred]
print(Y_pred_labels)

conf_mat = confusion_matrix(y_test, Y_pred_labels)

print(conf_mat)
plt.figure(figsize=(15,7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')

input_image_path = r'C:\Users\allen\OneDrive\Desktop\understanding\data\dl data\MNIST_digit.png'

input_image = cv2.imread(input_image_path)
print(type(input_image))
print(input_image)
cv2.imshow("Input Image", input_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
print(input_image.shape)

grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

print(grayscale.shape)
input_image_resize = cv2.resize(grayscale, (28, 28))
print(input_image_resize.shape)
cv2.imshow("Resized Image", input_image_resize)

cv2.waitKey(0)
cv2.destroyAllWindows()
input_image_resize = input_image_resize/255

print(type(input_image_resize))
image_reshaped = np.reshape(input_image_resize, [1,28,28])
input_prediction = model.predict(image_reshaped)
print(input_prediction)

input_pred_label = np.argmax(input_prediction)
print(input_pred_label)

input_image_path = input('Path of the image to be predicted: ')

input_image = cv2.imread(input_image_path)

cv2.imshow("Input Image", input_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

input_image_resize = cv2.resize(grayscale, (28, 28))

input_image_resize = input_image_resize/255

image_reshaped = np.reshape(input_image_resize, [1,28,28])

input_prediction = model.predict(image_reshaped)

input_pred_label = np.argmax(input_prediction)

print('The Handwritten Digit is recognised as ', input_pred_label)




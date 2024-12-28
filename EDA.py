import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.datasets import cifar10 

cifar =  cifar10
(training_images, training_labels), (testing_images, testing_labels) = cifar.load_data()


# shape of training dataset
print(training_images.shape) # (50000, 32, 32, 3)
print(training_labels.shape) # (50000, 1)

# shape of testing dataset
print(testing_images.shape) # (10000, 32, 32, 3)
print(testing_labels.shape) # (10000, 1)


# plotting a random set of images to see how the classes vary
fig = plt.figure(figsize=(12, 8))
columns = 5
rows = 3

for i in range(1, columns*rows +1):
   img = training_images[i] # get an image, defined as "img"
   fig.add_subplot(rows, columns, i) # create subplot (row index, col index, which number of plot)
   plt.title("Label:" + str(training_labels[i])) # plot the image, along with its label
   plt.imshow(img, cmap='binary')
plt.show()


# plotting a set of images from a single class to see how the images vary within one class
fig = plt.figure(figsize=(12, 8))
columns = 10
rows = 10
num = 500

counter = 1
for i in range(num):
  if training_labels[i] == 0:
      img = training_images[i] # get an image, defined as "img"
      fig.add_subplot(rows, columns, counter) # create subplot (row index, col index, which number of plot)
      plt.title("Label:" + str(training_labels[i])) # plot the image, along with its label
      plt.imshow(img, cmap='binary')
      counter += 1
plt.show()

# ensuring the dataset is balanced with 6,000 images in each class (5,000 in training and 1,000 in testing)
count = 0
counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

for i in training_labels:
  counts[int(i)] += 1

for i in testing_labels:
  counts[int(i)] += 1

print(counts)
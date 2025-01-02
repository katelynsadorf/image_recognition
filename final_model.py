import tensorflow as tf
from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Dense, Dropout, Activation, Flatten
from keras.datasets import cifar10
import matplotlib.pyplot as plt


# Architecture
#   Convolutional (32 filters, 3x3 kernel, 1x1 step)
#   Max pooling
#   Dropout (0.2)

#   Convolutional (64 filters, 3x3 kernel, 1x1 step)
#   Max pooling
#   Dropout (0.25)

#   Convolutional (128 filters, 3x3 kernel, 1x1 step)
#   Max pooling
#   Dropout (0.3)

#   Flatten

#   Dense (128)
#   Dense (64)
#   Dense (10) (softmax)


def gpu(x_train, y_train, epochs):
  '''Function to utilize gpu for training'''
  device_name = tf.test.gpu_device_name()
  if device_name != '/device:GPU:0': # if no gpu is found
    print('GPU device not found')
    history = model.fit(
        x_train, y_train,
        batch_size=256,
        epochs=epochs,
        validation_split=0.2,
        verbose=1)
    return history
  else: # else run with gpu
    print('Found GPU at: {}'.format(device_name))
    with tf.device('/device:GPU:0'):
      history = model.fit(
          x_train, y_train,
          batch_size=256,
          epochs=epochs,
          validation_split=0.2,
          verbose=1)
      return history

(training_images, training_labels), (testing_images, testing_labels) = cifar10.load_data()

model = Sequential()

# normalization
training_norm = training_images / 255.0
testing_norm = testing_images / 255.0


# Group 1
model.add(Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = "valid", activation = "relu", input_shape = (32, 32, 3)))
model.add(Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = "valid", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
model.add(Dropout(rate = 0.2))


# Group 2
model.add(Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "valid", activation = "relu"))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "valid", activation = "relu"))
model.add(MaxPooling2D(pool_size = (3, 3), strides = (1, 1)))
model.add(Dropout(rate = 0.2))


# Group 3
model.add(Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = "valid", activation = "relu"))
model.add(Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = "valid", activation = "relu"))
model.add(MaxPooling2D(pool_size = (4, 4), strides = (1, 1)))
model.add(Dropout(rate = 0.2))


# Flatten and dense layers
model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 10, activation = 'softmax'))


model.compile(
    loss = tf.keras.losses.sparse_categorical_crossentropy, # loss function
    optimizer = tf.keras.optimizers.Adam(), # optimizer function
    metrics = ['accuracy'] # reporting metric
)

history = gpu(training_norm, training_labels, 30)

plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True) # diagram
# ~75% validation accuracy
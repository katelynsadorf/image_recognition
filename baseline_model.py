import tensorflow as tf

from tensorflow.keras.datasets import cifar10 

cifar =  cifar10
(training_images, training_labels), (testing_images, testing_labels) = cifar.load_data()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(), # flatten turns the 28x28 shape into one long array of pixels (784 pixels long)
  tf.keras.layers.Dense(220, activation='relu'), # hidden layer
  tf.keras.layers.Dense(120, activation='relu'), # hidden layer
  tf.keras.layers.Dense(50, activation='relu'), # hidden layer
  tf.keras.layers.Dense(25, activation='relu'), # hidden layer
  tf.keras.layers.Dense(10, activation='softmax') # output layer
])

model.compile(optimizer = "adam", # recall: gradient descent, stochastic gradient descent, RmsProp, ADAM
              loss = 'sparse_categorical_crossentropy', # loss: mean_square_error, mean_absolute_error, cross_entropy, binary_cross_entropy
              metrics=['accuracy'])

history = model.fit(training_images, training_labels, epochs=30,
                    validation_data=[testing_images, testing_labels])
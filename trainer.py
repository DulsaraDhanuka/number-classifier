import tensorflow as tf
import pickle
import time

keras_models = tf.keras.models
keras_layers = tf.keras.layers
keras_callbacks = tf.keras.callbacks

NAME = "Digit-Classifier-64x2-CNN"

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

model = keras_models.Sequential()

model.add(keras_layers.Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(keras_layers.Activation('relu'))
model.add(keras_layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras_layers.Conv2D(64, (3, 3)))
model.add(keras_layers.Activation('relu'))
model.add(keras_layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras_layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(keras_layers.Dense(64, activation=tf.nn.relu))
model.add(keras_layers.Dense(64, activation=tf.nn.relu))

model.add(keras_layers.Dense(10, activation=tf.nn.softmax))

tensorboard = keras_callbacks.TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )

model.fit(X, y,
          epochs=3,
          callbacks=[tensorboard])
model.save(NAME + ".model")

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import cv2
import os

MODEL_PATH = "E:/number-classifier/Digit-Classifier-64x2-CNN.model"
IMG_PATH = "E:/number-classifier/image.png"
IMG_SIZE = 28

img_array = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_CUBIC)
predict_val = [img_array]
predict_val = np.array(predict_val).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model(MODEL_PATH)
prediction = model.predict(predict_val)
print(np.argmax(prediction))

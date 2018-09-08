import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import time

starttime = time.time()
print("Creating Training Data - " + str(starttime))
training_data = []
DATADIR = "E:/number-classifier/training_data"
CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
IMG_SIZE = 28

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            
create_training_data()

random.shuffle(training_data)

X = []
y = []

for features, labels in training_data:
    X.append(features)
    y.append(labels)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
print("Training Data Created Took " + str(time.time()-starttime) + " seconds")

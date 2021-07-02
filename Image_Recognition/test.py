import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2

import numpy as np

labels = ['blue', 'empty', 'yellow']
img_size = 58
filenames = []
def get_data(data_dir):
    data = [] 
    for img in os.listdir(data_dir):
        try:
            img_arr = cv2.imread(os.path.join(data_dir, img))[...,::-1] #convert BGR to RGB format
            filenames.append(img)
            resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
            data.append(resized_arr)
        except Exception as e:
            print(e)
    return np.array(data)

x_test = get_data("input/test")

model = tf.keras.models.load_model("model.h5")
prediction_matrices = model.predict(x_test)  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT

predictions = []
for i, prediction in enumerate(prediction_matrices):
    predictions.append(np.argmax(prediction))
    label = labels[np.argmax(prediction)]
    print(filenames[i] + " -> " + label)

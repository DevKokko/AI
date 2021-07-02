import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.core.fromnumeric import resize
from imageio import imread
import io
import base64

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 

labels = ['blue', 'empty','yellow']
img_size = 58

def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

def plot(train):
    l = []
    for i in train:
        if(i[1] == 0):
            l.append("blue")
        elif(i[1] == 1):
            l.append("yellow")
        else:
            l.append("empty")
    sns.set_style('darkgrid')
    sns.countplot(l)

    plt.figure(figsize = (5,5))
    plt.imshow(train[1][0])
    plt.title(labels[train[0][1]])

    plt.figure(figsize = (5,5))
    plt.imshow(train[-1][0])
    plt.title(labels[train[-1][1]])

def result_plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(history.history['val_loss']))

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def generate_model():
    model = Sequential()
    model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(img_size, img_size, 3)))
    model.add(MaxPool2D())

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    model.add(Dense(3, activation="softmax"))

    model.summary()
    return model

def decode_base64(base64data):
    converted_images = []
    test = ""
    for img in base64data:
        img_arr = imread(io.BytesIO(base64.b64decode(img)))#[...,::-1]
        test = img_arr
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB) 
        resized_arr = cv2.resize(img_arr, (img_size, img_size))
        converted_images.append(resized_arr)
    print(test)
    return np.array(converted_images)

def plot_spot(img, label):
    plt.figure(figsize = (5,5))
    plt.imshow(img)
    plt.title(label)
    plt.show()

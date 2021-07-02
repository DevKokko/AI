from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.utils as utils

from util import labels, img_size, get_data, plot, result_plot, generate_model

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf

import numpy as np

train = get_data('input/train')

plot(train)

data = []
labels = []
x_val = []
y_val = []

for feature, label in train:
  data.append(feature)
  labels.append(label)

x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.33, shuffle= True)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 0,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0, # Randomly zoom image 
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

datagen.fit(x_train)

model = generate_model()
utils.plot_model(model, to_file=f'model_schema.png', show_shapes=True, show_layer_names=False)

opt = Adam(lr=0.005)
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
history = model.fit(x_train,y_train,epochs = 50 , validation_data = (x_val, y_val), callbacks=[es])

result_plot(history)

predictions = model.predict_classes(x_val)
predictions = predictions.reshape(1,-1)[0]

model.save('model.h5')

print(classification_report(y_val, predictions, target_names = ['Blue (Class 0)','Yellow (Class 1)', 'Empty (Class 2)']))
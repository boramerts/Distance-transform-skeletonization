#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:01:25 2023

@author: boramert
"""

import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Dense, Activation, BatchNormalization, Dropout, Flatten, MaxPooling2D
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import sklearn.metrics

# Set parameters

# Skeleton image paths
testpath = "/Users/boramert/Desktop/Okul/BİL 587/Proje/DistanceTransform/Test"
trainpath = "/Users/boramert/Desktop/Okul/BİL 587/Proje/DistanceTransform/Train"

# Original image paths
#testpath = "/Users/boramert/Desktop/Okul/BİL 587/Proje/DATASET/Test"
#trainpath = "/Users/boramert/Desktop/Okul/BİL 587/Proje/DATASET/Train"

img_size = (224, 224)  # Set the desired image size
batch_size = 32  # Set the batch size for training and testing sets
num_classes = 5  # Set the number of classes in your dataset

traingen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.20
)

testgen = ImageDataGenerator(
    rescale=1./255
)


train_generator = traingen.flow_from_directory(
    trainpath,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    color_mode = 'grayscale'
)

validation_generator = traingen.flow_from_directory(
    trainpath,
    target_size = img_size,
    class_mode = 'categorical',
    subset = 'validation',
    shuffle = True,
    color_mode = 'grayscale')

test_generator = testgen.flow_from_directory(
    testpath,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    color_mode = 'grayscale'
)

# Print class labels
labels = {value: key for key, value in train_generator.class_indices.items()}
classes = []
print("Label Mappings for classes present in the training and validation datasets\n")
for key, value in labels.items():
    classes.append(value)
    print(str(key)," : ", str(value))

# Define the model architecture
model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation = 'relu', input_shape=(224, 224, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation = 'relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=128, kernel_size=(3, 3), activation = 'relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        
        Dense(units=128, activation='relu'),
        Dense(units=64, activation='relu'),
        Dropout(0.1),
        Dense(units=32, activation='relu'),
        Dense(units=5, activation='softmax')
])

model.summary()

# Reduce learning rate with validation loss
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=5)

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])

#Create a TensorBoard callback
logs = "logs_img/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 2,
                                                 profile_batch = 0)

history = model.fit(train_generator, epochs=50, validation_data=validation_generator,
                       verbose=2, # type: ignore
                       callbacks=[reduce_lr, tboard_callback])


train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

train_loss = history.history['loss']
val_loss = history.history['val_loss']

learning_rate = history.history['lr']

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))

ax[0].set_title('Training/Validation Accuracy vs. Epochs')
ax[0].plot(train_accuracy, 'o-', label='Training Accuracy')
ax[0].plot(val_accuracy, 'o-', label='Validation Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].legend(loc='best')

ax[1].set_title('Training/Validation Loss vs. Epochs')
ax[1].plot(train_loss, 'o-', label='Training Loss')
ax[1].plot(val_loss, 'o-', label='Validation Loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].legend(loc='best')

ax[2].set_title('Learning Rate vs. Epochs')
ax[2].plot(learning_rate, 'o-', label='Learning Rate')
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('Loss')
ax[2].legend(loc='best')

plt.tight_layout()
plt.show()

fig.savefig('Plots_DT.png', dpi=400)

print("Evaluate on test data")
results = model.evaluate(test_generator)
print("Test Accuracy: ", results[1])

true_labels = test_generator.classes
preds = model.predict(test_generator)
preds = np.array([np.argmax(x) for x in preds])

conf_matrix = sklearn.metrics.confusion_matrix(true_labels, preds) # type: ignore
fig = sklearn.metrics.ConfusionMatrixDisplay(conf_matrix).plot()

fig.figure_.savefig('Conf_matrix_DT.png', bbox_inches = 'tight', dpi = 400)

from tensorflow.keras.utils import plot_model
plot_model(model,show_layer_names=False,show_layer_activations=True,rankdir='LR',)



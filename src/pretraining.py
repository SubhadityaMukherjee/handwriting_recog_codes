# %%
import os
from PIL import Image, ImageFont, ImageDraw
from multiprocessing import Pool
from functools import partial
import numpy as np
from multiprocessing import Process, current_process
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import glob
from utils import *
import concurrent
import os
import time
import urllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

from tensorflow.keras.layers import *

"""
This is the training code for Dead Sea scrolls character recognition
"""

# %%
# Define defaults
# TODO: Change to args
main_path = Path("../data/")
dss_path = main_path / "monkbrill"
# print(dss_path)
batch_size = 200
image_size = (28, 28, 1)
AUTOTUNE = tf.data.AUTOTUNE

# Read data
images, labels = load_images_to_array(dss_path)
# print(images[0].shape)
labelmap, labels = label_to_dict(labels)
print(f"Total no of unique labels : {len(set(labels))}")
print(len(images))
# print(images[:3], labels[:3])
#%%
# Train test split
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=1337)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))
#%%
np.array(y_train).shape
#%%
# Convert to tf.data
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Prefetch data, shuffle, batch
train_dataset = train_dataset.shuffle(100).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# %%
#TODO Add data aug
# data_augmentation = keras.Sequential(
#     [

#     ]
# )

def make_model(input_shape, num_classes):
    """
    Just a placeholder classification model. Need to modify
    """
    model = tf.keras.Sequential([
        # keras.Input(shape=(None,28,28,1), name="image"),
        tf.keras.layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D((2, 2)),
        # tf.keras.layers.Flatten(128),
        # tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')

        # tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        # tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        # tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        # tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        # tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='valid'),
        # tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(64, activation='relu'),
        # tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dense(26, activation='softmax')
    ])
    return model



# model = make_model((28, 28), 27)

model = tf.keras.Sequential()
#convolutional layer with rectified linear unit activation
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
#32 convolution filters used each of size 3x3
#again
model.add(Conv2D(64, (3, 3), activation='relu'))
#64 convolution filters used each of size 3x3
#choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#randomly turn neurons on and off to improve convergence
model.add(Dropout(0.25))
#flatten since too many dimensions, we only want a classification output
model.add(Flatten())
#fully connected to get all relevant data
model.add(Dense(128, activation='relu'))
#one more dropout for convergence' sake :) 
model.add(Dropout(0.5))
#output a softmax to squash the matrix into output probabilities
model.add(Dense(27, activation='softmax'))

print(model.summary())

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[
                  tf.keras.metrics.SparseCategoricalAccuracy()
])

model.fit(train_dataset, epochs=20, callbacks=[
    tf.keras.callbacks.ModelCheckpoint("./logs/save_at_{epoch}.h5"),
],
)

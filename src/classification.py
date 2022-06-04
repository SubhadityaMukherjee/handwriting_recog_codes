import concurrent
import glob
import os
import time
import urllib
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Pool, Process, current_process
from pathlib import Path
from types import SimpleNamespace
from typing import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from tensorflow import keras
# from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import Model, applications, layers
from tensorflow.keras.layers import (Activation, Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D)
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras import backend as K

from utils import *


# Make model
def make_model():
    model = tf.keras.Sequential(
        [
            # convolutional layer with rectified linear unit activation
            # kernel size used to be 3, 3
            tf.keras.layers.Conv2D(
                32, kernel_size=(5, 5), activation="relu", input_shape=(28, 28, 1)
            ),
            # tf.keras.layers.BatchNormalization(), #check this at home
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            # tf.keras.layers.BatchNormalization(), #check this at home
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.25),  # decreased from 0.5 to 0.25
            tf.keras.layers.Dense(27, activation="softmax"),
        ]
    )
    return model


labelmap = {'Alef': 0, 'Ayin': 1, 'Bet': 2, 'Dalet': 3, 'Gimel': 4, 'He': 5, 'Het': 6, 'Kaf': 7, 'Kaf-final': 8, 'Lamed': 9, 'Mem': 10, 'Mem-medial': 11, 'Nun-final': 12, 'Nun-medial': 13, 'Pe': 14, 'Pe-final': 15, 'Qof': 16, 'Resh': 17, 'Samekh': 18, 'Shin': 19, 'Taw': 20, 'Tet': 21, 'Tsadi-final': 22, 'Tsadi-medial': 23, 'Waw': 24, 'Yod': 25, 'Zayin': 26}

# load model 
model = make_model()
model.load_weights("trained_model.h5")

# load data to predict on
# load monkbrill data
# TODO: change to actual data lol

main_path = Path("lines/P22-Fg008-R-C01-R01-binarized.jpg")
dss_path = main_path / "characters"
# print(dss_path)
batch_size = 200
image_size = (28, 28, 1)
AUTOTUNE = tf.data.AUTOTUNE

# Read data
images, labels = load_images_to_array_no_label(dss_path)

#Convert to tf.data
train_dataset = tf.data.Dataset.from_tensor_slices((images, labels))

# # Prefetch data, shuffle, batch
train_dataset = train_dataset.batch(batch_size)

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

# predict (either in batches or not? not sure)
predictions = model.predict(train_dataset)

#%%
# get key
def get_key(val):
    for key, value in labelmap.items():
        if int(val) is int(value):
            return key
    return "Key not found"


classes = np.argmax(predictions, axis=1)

# print prediction to file 
with open("prediction.txt", "w") as f:
    for value in classes:
        f.write(get_key(value) + "\n")

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

"""
This is the training code
"""

# %%
# Define defaults
# TODO: Change to args
main_path = Path("/media/hdd/github/handwriting_recog_codes/data/")
dss_path = main_path / "monkbrill"
batch_size = 200
image_size = (28, 28, 1)
AUTOTUNE = tf.data.AUTOTUNE

# Read data
images, labels = load_images_to_array(dss_path)
labelmap, labels = label_to_dict(labels)
print(f"Total no of unique labels : {len(set(labels))}")
print(len(images))
print(images[:3], labels[:3])
# Train test split
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=1337)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

# Convert to tf.data
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# Prefetch data, shuffle, batch
train_dataset = train_dataset.shuffle(100).batch(batch_size).cache().prefetch(AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).cache().prefetch(AUTOTUNE)
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
        # data_augmentation,
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


model = make_model((28, 28), 27)

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[
                  tf.keras.metrics.SparseCategoricalAccuracy()
])
model.fit(train_dataset, epochs=20, callbacks=[
    tf.keras.callbacks.ModelCheckpoint("./logs/save_at_{epoch}.h5"),
],
)

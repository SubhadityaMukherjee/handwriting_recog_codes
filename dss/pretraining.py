# %%
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
from tensorflow.python.keras import backend as K
from utils import *




"""
This is the training code for Dead Sea scrolls character recognition
"""

# Define functions

# Make model
def make_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, kernel_size=(5, 5), activation="relu", input_shape=(28, 28, 1)
            ),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.25),  # decreased from 0.5 to 0.25
            tf.keras.layers.Dense(27, activation="softmax"),
        ]
    )
    return model

# make plots 
def make_plots(history):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig("results/dss_classification_accuracy.png", dpi=300)
    plt.clf()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig("results/dss_classification_loss.png", dpi=300)


# %%
## PRETRAINING 

# Define defaults
main_path = Path(".")
dss_path = main_path / "new_data"
batch_size = 200
image_size = (28, 28, 1)
AUTOTUNE = tf.data.AUTOTUNE

# Read data
images, labels = load_images_to_array_png(dss_path)
labelmap, labels = label_to_dict(labels)
print(f"Total no of unique labels : {len(set(labels))}")
print(len(images))

#%%
# Train test split
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=1337
)
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

#%%
# Build model

model = make_model()

print(model.summary())

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=20,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint("./logs/save_at_{epoch}.h5"),
    ],
)

# save model 
model.save_weights("models/pretrained_model.h5")

# %%
## TRAINING on monkbrill data

# define defaults 
main_path = Path("../data/")
dss_path = main_path / "monkbrill"
batch_size = 200
image_size = (28, 28, 1)
AUTOTUNE = tf.data.AUTOTUNE

# Read data
images, labels = load_images_to_array(dss_path)
labelmap, labels = label_to_dict(labels)

print(f"Total no of unique labels : {len(set(labels))}")
print(len(images))

# Train test split
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=1337
)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

np.array(y_train).shape

# Convert to tf.data
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Prefetch data, shuffle, batch
train_dataset = train_dataset.shuffle(100).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# make and fit model
model = make_model()

model.load_weights("models/pretrained_model.h5")

print(model.summary())

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=20,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint("./logs/save_at_{epoch}.h5"),
    ],
)

# save model 
model.save("models/trained_model.h5")

make_plots(history)
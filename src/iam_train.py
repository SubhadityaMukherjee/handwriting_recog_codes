# %%
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
from utils import *
from models import *

from pathlib import Path

np.random.seed(42)
tf.random.set_seed(42)

AUTOTUNE = tf.data.AUTOTUNE
# %%
# Define defaults
# TODO: Change to args
main_path = Path("../data/IAM-data/")
images_path = main_path / "img"
labels_path = main_path / "iam_lines_gt.txt"
batch_size = 200
padding_token = 99
image_width = 128
image_height = 32
image_size = (image_width, image_height)

AUTOTUNE = tf.data.AUTOTUNE
# %%
# Read data
images, labels = iam_data_reader(
    images_path, labels_path, (image_width, image_height), subset=2001)
print(images[:3], labels[:3])
print(len(images), len(labels))

# Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=1337)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

# %%
# Get vocabulary size
train_labels_cleaned, max_len, characters = vocabulary_size(y_train)
test_labels_cleaned = clean_labels(y_test)
train_labels_cleaned[:10]
# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)
# %%
# store params so it can be used between multiple functions
sprocess = StoreAndProcess()
sprocess.store_params(
    {
        "batch_size": batch_size,
        "padding_token": padding_token,
        "image_size": image_size,
        "image_width": image_width,
        "image_height": image_height,
        "max_len": max_len,
        "characters": characters,
        "char_to_num": char_to_num,
        "num_to_char": num_to_char,
    }

)
# %%
train_ds = sprocess.prepare_dataset(x_train, train_labels_cleaned)
test_ds = sprocess.prepare_dataset(x_test, test_labels_cleaned)
# %%
train_ds.take(1).take(1)
# %%
sprocess.view_batch(train_ds)
# %%
model = simple_iam(sprocess.params)
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)

# model.summary()
# %%
validation_images = []
validation_labels = []

for batch in test_ds:
    validation_images.append(batch["image"])
    validation_labels.append(batch["label"])

# %%
epochs = 10  # To get good results this should be at least 50.

edit_distance_callback = EditDistanceCallback(prediction_model)

# Train the model.
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs,
    callbacks=[edit_distance_callback],
)


# %%

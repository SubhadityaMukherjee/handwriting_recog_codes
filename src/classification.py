import concurrent
import glob
import os
import time
import shutil
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
# font = ImageFont.truetype(
#     "Habbakuk.TTF", 42
# )  

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


def get_key(val):
    for key, value in labelmap.items():
        if int(val) is int(value):
            return key
    return "Key not found"


labelmap = {'Alef': 0, 'Ayin': 1, 'Bet': 2, 'Dalet': 3, 'Gimel': 4, 'He': 5, 'Het': 6, 'Kaf': 7, 'Kaf-final': 8, 'Lamed': 9, 'Mem': 10, 'Mem-medial': 11, 'Nun-final': 12, 'Nun-medial': 13,
            'Pe': 14, 'Pe-final': 15, 'Qof': 16, 'Resh': 17, 'Samekh': 18, 'Shin': 19, 'Taw': 20, 'Tet': 21, 'Tsadi-final': 22, 'Tsadi-medial': 23, 'Waw': 24, 'Yod': 25, 'Zayin': 26}


# load model
model = make_model()
try:
    model.load_weights("models/trained_model.h5")
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

except:
    print("No model found")
    exit(0)
try:
    shutil.rmtree("results/classification")
except:
    pass
os.makedirs("results/classification", exist_ok=True)


def load_images_char_class(linespath):
    """
    Predict the character class of each file, combines them into lines and save the results to a file.
    """
    # ims, labels = [], []
    for root in tqdm(os.listdir(linespath)):
        lines = os.listdir(os.path.join(linespath, root, "characters"))
        extracted = []
        for line in lines:
            images = []
            labels = []
            for char in os.listdir(os.path.join(linespath, root, "characters", line)):
                char = os.path.join(linespath, root, "characters", line, char)
                im = np.array(Image.open(char).convert("L").resize(
                    (28, 28), Image.Resampling.BILINEAR))
                im = im[..., np.newaxis]
                im = im / 255.0
                images.append(im)
                labels.append(char +" ")

            train_dataset = tf.data.Dataset.from_tensor_slices(
                (images, labels))

            train_dataset = train_dataset.batch(1)

            # predict (either in batches or not? not sure)
            try:
                predictions = model.predict(train_dataset)
                classes = np.argmax(predictions, axis=1)
                classes = [get_key(x) for x in classes]
                extracted.append(" ".join(classes))

            except:
                print("Error predicting")
                continue

        with open(os.path.join("results/classification", root + ".txt"), "w") as f:
            f.write("\n".join(extracted))


main_path = Path("lines")
image_size = (28, 28, 1)
AUTOTUNE = tf.data.AUTOTUNE

# Read data
load_images_char_class(main_path)

# # Convert to tf.data
# train_dataset = tf.data.Dataset.from_tensor_slices((images, labels))

# # # Prefetch data, shuffle, batch
# train_dataset = train_dataset.batch(batch_size)

# model.compile(
#     optimizer=tf.keras.optimizers.RMSprop(),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=["accuracy"],
# )

# # predict (either in batches or not? not sure)
# predictions = model.predict(train_dataset)

# classes = np.argmax(predictions, axis=1)

# # print prediction to file
# with open("prediction.txt", "w") as f:
#     for value in classes:
#         f.write(get_key(value) + "\n")

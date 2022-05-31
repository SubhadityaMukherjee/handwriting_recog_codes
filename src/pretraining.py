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
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import (
    Activation,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
)
# from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import applications
from tensorflow.python.keras import backend as K 
from tensorflow.keras import Model
# import hiddenlayer as hl
# import hiddenlayer.transforms as ht


from tensorflow.keras.layers import *

"""
This is the training code for Dead Sea scrolls character recognition
"""

# %%
# Define defaults
# TODO: Change to args
main_path = Path("data/")
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
# brightness, elastic transform, shear, scale, gaussian blur, dilate, erode 
# data_augmentation = keras.Sequential(
#     [
#     ]
# )

def make_model(model_type):
    if model_type == "CNN":
        model = tf.keras.Sequential([
            #convolutional layer with rectified linear unit activation
            # kernel size used to be 3, 3
            tf.keras.layers.Conv2D(32, kernel_size=(5, 5),
                    activation='relu',
                    input_shape=(28,28,1)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(27, activation='softmax') 
        ])
    elif model_type == "ResNet":
        model=applications.ResNet50(weights=None, include_top=False)
        x = model.output
        #x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(131, activation='softmax')(x)
        model = Model(inputs=model.input, outputs=predictions)
    else: 
        pass
    return model


model = make_model("CNN")

print(model.summary())
plot_model(model, to_file="model.png", show_shapes=False)

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[
                  'accuracy'
])

# TODO: add validation split? 
history = model.fit(train_dataset, validation_data=test_dataset, epochs=20, callbacks=[
    tf.keras.callbacks.ModelCheckpoint("./logs/save_at_{epoch}.h5"),
],
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('accuracy.png', dpi=300)
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss.png', dpi=300)



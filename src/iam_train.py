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
main_path = Path("data/IAM-data/")
images_path = main_path / "img"
labels_path = main_path / "iam_lines_gt.txt"
batch_size = 64
padding_token = 99
image_width = 128
image_height = 32
image_size = (image_width, image_height)

AUTOTUNE = tf.data.AUTOTUNE
# %%
# Read data
images, labels = iam_data_reader(
    images_path, labels_path, (image_width, image_height), subset=2001)
# print(images[:3], labels[:3])
print(len(images), len(labels))

# Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=1337)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

# %%
# Get vocabulary size
train_labels_cleaned, max_len, characters = vocabulary_size(y_train)
test_labels_cleaned, _, _ = vocabulary_size(y_test)
# print(train_labels_cleaned[:10])
# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)
#%%
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


def calculate_edit_distance(labels, predictions):
    # Get a single batch and convert its labels to sparse tensors.
    saprse_labels = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

    # Make predictions and convert them to sparse tensors.
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    predictions_decoded = keras.backend.ctc_decode(
        predictions, input_length=input_len, greedy=True
    )[0][0][:, :max_len]
    sparse_predictions = tf.cast(
        tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
    )

    # Compute individual edit distances and average them out.
    edit_distances = tf.edit_distance(
        sparse_predictions, saprse_labels, normalize=False
    )
    return tf.reduce_mean(edit_distances)


class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, pred_model):
        super().__init__()
        self.prediction_model = pred_model

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []

        for i in range(len(validation_images)):
            labels = validation_labels[i]
            predictions = self.prediction_model.predict(validation_images[i])
            edit_distances.append(calculate_edit_distance(
                labels, predictions).numpy())

        print(
            f"Edit {epoch + 1}: {np.mean(edit_distances):.4f}"
        )
#%%
model.summary()

# %%
epochs = 50  # To get good results this should be at least 50.

edit_distance_callback = EditDistanceCallback(prediction_model)

# Train the model.
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs,
    # callbacks=[edit_distance_callback],
)


# %%
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


#  Let's check results on some test samples.
for batch in test_ds.take(1):
    batch_images = batch["image"]
    _, ax = plt.subplots(4, 4, figsize=(15, 8))

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    for i in range(16):
        img = batch_images[i]
        img = tf.image.flip_left_right(img)
        img = tf.transpose(img, perm=[1, 0, 2])
        img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
        img = img[:, :, 0]

        title = f"Prediction: {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")

plt.show()

# %%

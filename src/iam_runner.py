#%%
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
#%%
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

#%%
def vocabulary_size(y_train):
    """
    Find maximum length and the size of the vocabulary in the training data.
    """
    train_labels_cleaned = []
    characters = set()
    chars = []
    max_len = 0

    for label in y_train:
        for char in label:
            characters.add(char)
            chars.append(char)
        label = " ".join(clean_labels(label))

        train_labels_cleaned.append(label)
        max_len = max(max_len, len(chars))
        chars = []
    print(train_labels_cleaned[:3])

    print("Maximum length: ", max_len)
    print("Vocab size: ", len(characters))
    return train_labels_cleaned, max_len, characters

# %%
# Get vocabulary size
train_labels_cleaned, max_len, characters = vocabulary_size(y_train)
test_labels_cleaned, _, _ = vocabulary_size(y_test)
# print(train_labels_cleaned[:10])
# train_labels_cleaned = [x[:max_len] for x in train_labels_cleaned]
# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)
#%%
# max([len(x) for x in train_labels_cleaned])
#%%
batch_size = 64
padding_token = 99
image_width = 128
image_height = 32

def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label


def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)
#%%
train_ds = prepare_dataset(x_train, train_labels_cleaned)
validation_ds = prepare_dataset(x_test, test_labels_cleaned)
#%%
for data in train_ds.take(1):
    images, labels = data["image"], data["label"]

    _, ax = plt.subplots(4, 4, figsize=(15, 8))

    for i in range(16):
        img = images[i]
        img = tf.image.flip_left_right(img)
        img = tf.transpose(img, perm=[1, 0, 2])
        img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
        img = img[:, :, 0]

        # Gather indices where label!= padding_token.
        label = labels[i]
        indices = tf.gather(label, tf.where(tf.math.not_equal(label, padding_token)))
        # Convert to string.
        label = tf.strings.reduce_join(num_to_char(indices))
        label = label.numpy().decode("utf-8")

        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")


plt.show()

# %%
class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost
        # self.loss_fn = tf.compat.v1.nn.ctc_loss

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        # y_true = tf.keras.backend.ctc_label_dense_to_sparse(y_true,label_length)
        # y_pred = tf.keras.backend.ctc_label_dense_to_sparse(y_pred,label_length)
        # y_pred = y_pred[:, 2:, :]
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred

#%%
from tensorflow.keras import layers
def build_model(image_width, image_height, dropout): # Builiding the model
    # Inputs to the model Keras
    input_img = keras.Input(shape=(image_width, image_height, 1), name="image")
    labels = layers.Input(name="label", shape=(None,))

    # CNN options
    cnn = 8 # number of CNN layers
    if cnn==5:
        kernel_vals = [5,5,3,3,3]
        feature_vals = [32,64,128,128,256]
        pool_vals = [(2,2),(2,2),(1,2),(1,2),(1,2)]
        conv_names = ["Conv1","Conv2","Conv3","Conv4","Conv5",]
        pool_names = ["Pool1","Pool2","Pool3","Pool4","Pool5",]
        dropout_vals = [0, 0, 0, 0, 0]
    if cnn == 8:
        kernel_vals = [5,5,5,5,3,3,3,3]
        feature_vals = [32,32,64,64,128,128,128,256]
        pool_vals = [(1,1),(2,2),(1,1),(2,2),(1,2),(1,1),(1,2),(1,2)]
        conv_names = ["Conv1","Conv2","Conv3","Conv4","Conv5","Conv6","Conv7","Conv8",]
        pool_names = ["Pool1","Pool2","Pool3","Pool4","Pool5","Pool6","Pool7","Pool8",]
        # dropout_vals = [0, 0, 0, 0, 0.1, 0.15, 0.2, 0.2]
        dropout_vals = [0, 0, 0, 0, 0, 0, 0, 0]
    num_convs = len(kernel_vals)

    x = input_img
    for i in range(num_convs):
        x = layers.Conv2D(
        feature_vals[i],
        kernel_vals[i],
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name=conv_names[i],
        )(x)
        x = layers.MaxPooling2D(pool_vals[i], name=pool_names[i])(x)
        x = layers.Dropout(dropout_vals[i])(x)
    
    x = layers.Reshape(target_shape=(32,256), name="reshape1")(x)
    # x = layers.Dropout(0.33)(x)

    # RNN layers
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=dropout))(x)
    # x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25))(x)

    x = layers.Reshape(target_shape=(1,32,512), name="reshape2")(x)

    x = layers.Conv2D(len(char_to_num.get_vocabulary()) + 1,1, activation="softmax", name="conv_last")(x)
    x = layers.Reshape(target_shape=(32,len(char_to_num.get_vocabulary()) + 1), name="reshape3")(x) # Fix names for get layers? 

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Defining the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    
    # Optimizer
    opt = keras.optimizers.Adam()
    
    # Compile the model and return
    model.compile(optimizer=opt)
    return model
#%%
model = build_model(image_width, image_height, dropout=0.25)
model.summary()
# %%
validation_images = []
validation_labels = []

for batch in validation_ds:
    validation_images.append(batch["image"])
    validation_labels.append(batch["label"])
    # break

# %%
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
            edit_distances.append(calculate_edit_distance(labels, predictions).numpy())

        print(
            f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
        )

# %%
epochs = 10  # To get good results this should be at least 50.

prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="reshape3").output
)
edit_distance_callback = EditDistanceCallback(prediction_model)

# Train the model.
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs,
    callbacks=[edit_distance_callback],
)

# %%

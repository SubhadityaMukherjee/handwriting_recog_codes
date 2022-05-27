# %%
from cmath import inf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
# from utils import *
from models import *

from pathlib import Path

np.random.seed(42)
tf.random.set_seed(42)

AUTOTUNE = tf.data.AUTOTUNE
# %%
# Define defaults
# TODO: Change to args
main_path = Path("../../data/IAM-data/")
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
def iam_data_reader(images_path, labels_path, subset=None):
        """
        Reads the IAM dataset and returns images and labels
        """
        images, labels = [], []

        with open(labels_path, "r") as f:
            full_text = f.readlines()
            if subset:
                full_text = full_text[:subset]
            for line in tqdm(full_text, total=len(full_text)):
                if "png" in line:
                    fname = os.path.join(images_path, line.strip())
                    images.append(fname)
                elif len(line) > 1:
                    string_encode = line.lower().strip().encode("ascii", "ignore")
                    string_decode = string_encode.decode()
                    labels.append(string_decode)
                else:
                    continue
        return images, labels

images, labels = iam_data_reader(
    images_path, labels_path, subset=2001)
# print(images[:3], labels[:3])
print(len(images), len(labels))
#%%
# Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=1337)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

# %%
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
        label = clean_labels(label)

        train_labels_cleaned.append(label)
        max_len = max(max_len, len(chars))
        chars = []
    print(train_labels_cleaned[:3])

    print("Maximum length: ", max_len)
    print("Vocab size: ", len(characters))
    return train_labels_cleaned, max_len, characters


def clean_labels(labels):
    """
    Clean the labels and return a list of cleaned labels.
    """
    return [x.strip() for x in labels.split(" ")]


def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

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

class StoreAndProcess:
    """
    A class to store and process the images and labels
    """

    def __init__(self):
        self.params = {}

    def store_params(self, params):
        self.params = params

    def get_params(self):
        return self.params

    def load_and_preprocess_image_iam(self, image):
        image = tf.io.read_file(image)
        image = tf.image.decode_png(image, 3)
        image = distortion_free_resize(image, self.params["image_size"])
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def prepare_image(self, img):
        # img = tf.keras.preprocessing.image.load_img('iam_database/iam_database_formsA-D/a01-000u.png')

        a = tf.keras.preprocessing.image.img_to_array(img)
        h, w, _ = a.shape

        x = a.reshape(w,h, 3)

        return x // 255

    def prepare_image_reverse(self, img):
        # img = tf.keras.preprocessing.image.load_img('iam_database/iam_database_formsA-D/a01-000u.png')

        a = tf.keras.preprocessing.image.img_to_array(img)
        h, w, _ = a.shape

        # x = a.reshape(w, h)
        x = np.expand_dims(a, -1)

        return x // 255


    def vectorize_label(self, label):
        label = self.params["char_to_num"](tf.strings.unicode_split(
            label, input_encoding="UTF-8"))

        # print(label)
        length = tf.shape(label)[0]
        pad_amount = self.params["max_len"] - length
        # if length < 77:
        #     pad_amount = 77 - length
        # else:
        #     pad_amount = 0
        # pad_amount = 200
        label = tf.pad(label, paddings=[[0, pad_amount]],
                       constant_values=self.params["padding_token"])
        return label

    def prepare_dataset(self, image_paths, labels):
        labels = list(map(lambda x: self.vectorize_label(" ".join(x)), labels))
        image_paths = list(map(lambda x: self.prepare_image(self.load_and_preprocess_image_iam(x)), image_paths))
        dict_k = {"image": image_paths, "label": labels, "input_1": image_paths, "the_labels": labels, "label_length": [len(x) for x in labels], "input_length": [len(x) for x in labels]}

        dataset = tf.data.Dataset.from_tensor_slices(dict_k)
        # dataset.batch_size = self.params["batch_size"]
        # dataset.size = len(image_paths)
        # dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        # self.process_images_labels, num_parallel_calls=tf.data.AUTOTUNE
        # )
        return dataset.batch(self.params["batch_size"])
        # return dataset

    def view_batch(self, train_ds): #todo
        """
        View a batch of images and labels.
        """
        for data in train_ds.take(1):
            images, labels = data["image"], data["label"]

            _, ax = plt.subplots(4, 4, figsize=(15, 8))

            for i in range(16):
                img = images[i]
                img = self.prepare_image_reverse(img)
                img = tf.image.flip_left_right(img)
                img = tf.transpose(img, perm=[1, 0, 2])
                img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
                img = img[:, :, 0]
                # img = img.expand_dims(0)

                # Gather indices where label!= padding_token.
                label = labels[i]
                print(label)
                indices = tf.gather(label, tf.where(
                    tf.math.not_equal(label, self.params["padding_token"])))
                # Convert to string.
                label = tf.strings.reduce_join(
                    self.params["num_to_char"](indices))
                label = label.numpy().decode("utf-8")

                ax[i // 4, i % 4].imshow(img, cmap="gray")
                ax[i // 4, i % 4].set_title(label)
                ax[i // 4, i % 4].axis("off")
        # plt.show()
        # plt.savefig("./outputs/iam_train_batch.pdf")
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
# sprocess.view_batch(train_ds)
# %%
model_main = CtcModel(64, len(char_to_num.get_vocabulary()), image_height)
# train_model = model_main._create_training_model()
inference_model = model_main._create_inference_model()

# model.summary()
#%%
# train_model.summary()

# %%
epochs = 1  # To get good results this should be at least 50.

# edit_distance_callback = EditDistanceCallback(inference_model)
# train_model.compile()
history = model_main.fit(
    train_ds, test_ds
    )

#%%
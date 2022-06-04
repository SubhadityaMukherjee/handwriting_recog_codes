import concurrent
import os
import time
import urllib
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob
from pathlib import Path
from types import SimpleNamespace
from typing import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tqdm import tqdm

# from IPython.display import Image


"""
This module contains all the general add ons
"""


def load_images_to_array(dss_path):
    """
    Return images and labels as numpy arrays
    """
    images, labels = [], []
    for root, dirs, files in tqdm(os.walk(dss_path)):
        for file in files:
            if file.endswith(".pgm"):
                fname = os.path.join(root, file)
                # print(fname)
                im = np.array(
                    Image.open(fname)
                    .convert("L")
                    .resize((28, 28), Image.Resampling.BILINEAR)
                )
                im = im[..., np.newaxis]
                # print(im.shape)
                images.append(im)
                # get the last folder name as label
                labels.append(root.split("/")[-1])
    return images, labels


def load_images_to_array_png(dss_path):
    """
    Return images and labels as numpy arrays
    """
    images, labels = [], []
    for root, dirs, files in tqdm(os.walk(dss_path)):
        for file in files:
            if file.endswith(".png"):
                fname = os.path.join(root, file)
                # print(fname)
                im = np.array(
                    Image.open(fname)
                    .convert("L")
                    .resize((28, 28), Image.Resampling.BILINEAR)
                )
                im = im[..., np.newaxis]
                # print(im.shape)
                images.append(im)
                # get the last folder name as label
                labels.append(root.split("/")[-1])
    return images, labels


def label_to_dict(labels):
    labelmap = {label: i for i, label in enumerate(np.unique(labels))}
    return labelmap, [labelmap[label] for label in labels]


def iam_data_reader(images_path, labels_path, image_size, subset=None):
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
                # images.append(np.array(Image.open(fname).convert("L").resize(image_size, Image.Resampling.BILINEAR)))
                images.append(fname)
            elif len(line) > 1:
                # labels.append(line.strip())
                string_encode = line.lower().strip().encode("ascii", "ignore")
                string_decode = string_encode.decode()
                labels.append(string_decode)

                # print(line)
            else:
                continue
    return images, labels


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


"""
Preprocessing for images in DeadSea Scrolls
"""


def preprocess(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernelSize = 4
    maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))

    morphErode = cv2.morphologyEx(grayscale, cv2.MORPH_ERODE, maxKernel)

    kernelSize = 7
    maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))

    morphClose = cv2.morphologyEx(morphErode, cv2.MORPH_CLOSE, maxKernel)

    kernelSize = 3
    maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))

    morphOpen = cv2.morphologyEx(morphErode, cv2.MORPH_OPEN, maxKernel)

    # binary = cv2.threshold(morphOpen, 127, 255, cv2.THRESH_BINARY)

    return morphErode


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
        image = tf.image.decode_png(image, 1)
        image = distortion_free_resize(image, self.params["image_size"])
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def vectorize_label(self, label):
        label = self.params["char_to_num"](
            tf.strings.unicode_split(label, input_encoding="UTF-8")
        )

        # print(label)
        length = tf.shape(label)[0]
        pad_amount = self.params["max_len"] - length
        # if length < 77:
        #     pad_amount = 77 - length
        # else:
        #     pad_amount = 0
        # pad_amount = 200
        label = tf.pad(
            label,
            paddings=[[0, pad_amount]],
            constant_values=self.params["padding_token"],
        )
        return label

    def prepare_dataset(self, image_paths, labels):
        labels = list(map(lambda x: self.vectorize_label(" ".join(x)), labels))
        image_paths = list(
            map(lambda x: self.load_and_preprocess_image_iam(x), image_paths)
        )
        dict_k = {"image": image_paths, "label": labels}

        dataset = tf.data.Dataset.from_tensor_slices(dict_k)
        # dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        # self.process_images_labels, num_parallel_calls=tf.data.AUTOTUNE
        # )
        return dataset.batch(self.params["batch_size"])

    def view_batch(self, train_ds):
        """
        View a batch of images and labels.
        """
        for data in train_ds.take(1):
            images, labels = data["image"], data["label"]

            _, ax = plt.subplots(4, 4, figsize=(15, 8))

            for i in range(16):
                img = images[i]
                # img = img.expand_dims(0)
                # img = tf.expand_dims(img, -1)
                img = tf.image.flip_left_right(img)
                img = tf.transpose(img, perm=[1, 0, 2])
                img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
                img = img[:, :, 0]
                # img = img.expand_dims(0)

                # Gather indices where label!= padding_token.
                label = labels[i]
                print(label)
                indices = tf.gather(
                    label,
                    tf.where(tf.math.not_equal(label, self.params["padding_token"])),
                )
                # Convert to string.
                label = tf.strings.reduce_join(self.params["num_to_char"](indices))
                label = label.numpy().decode("utf-8")

                ax[i // 4, i % 4].imshow(img, cmap="gray")
                ax[i // 4, i % 4].set_title(label)
                ax[i // 4, i % 4].axis("off")
        # plt.show()
        plt.savefig("./outputs/iam_train_batch.pdf")

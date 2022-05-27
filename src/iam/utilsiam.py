import json
import logging
import math
import os
import random
import subprocess
from importlib import import_module
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import ParseError

import networkx as nx
import numpy as np
import scipy
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm


class CerCallback(Callback):
    def __init__(self, char_table, train_gen, val_gen, model, steps=None, interval=10):
        super().__init__()
        self._char_table = char_table
        self._train_gen = train_gen
        self._val_gen = val_gen
        self._model = model
        self._steps = steps
        self._interval = interval

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self._interval == 0 and epoch > 0:
            train_cer = self.compute_ler(self._train_gen)
            val_cer = self.compute_ler(self._val_gen)
            print("train LER {}; val LER {}".format(train_cer, val_cer))

    def compute_ler(self, gen):
        cer = LEREvaluator(self._model, gen, self._steps, self._char_table)
        return cer.evaluate()


class MyModelCheckpoint(Callback):
    def __init__(self, model, save_path, preprocessing_params):
        super().__init__()
        self._model = model
        self._save_path = save_path
        self._preprocessing_params = preprocessing_params

    def on_epoch_end(self, epoch, logs=None):
        self._model.save(self._save_path, self._preprocessing_params)

def to_sparse_tensor(sequences):
    indices = []
    values = []
    max_len = max(len(s) for s in sequences)

    for i in range(len(sequences)):
        for j in range(len(sequences[i])):
            indices.append((i, j))
            values.append(sequences[i][j])

    dense_shape = [len(sequences), max_len]
    return tf.SparseTensor(indices, values, dense_shape)


def fill_gaps(labels):
    for y in labels:
        if len(y) == 0:
            y.append(-1)


def compute_edit_distance(y_true, y_pred, normalize=True):
    fill_gaps(y_pred)

    sparse_y_true = to_sparse_tensor(y_true)
    sparse_y_pred = to_sparse_tensor(y_pred)
    return tf.edit_distance(sparse_y_pred, sparse_y_true, normalize=normalize)


def compute_cer(y_true, y_pred):
    distances = compute_edit_distance(y_true, y_pred, normalize=False)
    return normalize_distances(distances, y_true, y_pred)


def normalize_distances(distances, expected_labels, predicted_labels):
    norm_factors = []
    for i, dist in enumerate(distances):
        max_len = max(len(expected_labels[i]), len(predicted_labels[i]))
        norm_factors.append(max_len)

    return tf.divide(
        tf.dtypes.cast(distances, tf.float32),
        tf.constant(norm_factors, dtype=tf.float32),
    )


def codes_to_string(codes, char_table):
    return "".join([char_table.get_character(code) for code in codes])


def get_meta_info(path="lines_dataset/train"):
    import json

    meta_path = os.path.join(path, "meta.json")
    with open(meta_path) as f:
        s = f.read()

    meta_info = json.loads(s)
    return meta_info


def get_meta_attribute(path, attr_name):
    return get_meta_info(path)[attr_name]


def get_max_image_height(root_path):
    train = os.path.join(root_path, "train")
    val = os.path.join(root_path, "validation")
    test = os.path.join(root_path, "test")

    train_height = get_meta_attribute(train, "max_height")
    val_height = get_meta_attribute(val, "max_height")
    test_height = get_meta_attribute(test, "max_height")

    return max(train_height, val_height, test_height)


def make_true_labels(labels, label_lengths):
    flat_label_lengths = label_lengths.flatten()
    expected_labels = []

    for i, labelling in enumerate(labels):
        labelling_len = flat_label_lengths[i]
        expected_labels.append(labelling.tolist()[:labelling_len])

    return expected_labels


def cer_on_batch(model, batch):
    X, labels, input_lengths, label_lengths = batch

    predicted_labels = model.predict(X, input_lengths=input_lengths).tolist()

    expected_labels = make_true_labels(labels, label_lengths)

    label_error_rates = compute_cer(expected_labels, predicted_labels)

    return tf.reduce_mean(label_error_rates).numpy().flatten()[0]


class LEREvaluator:
    def __init__(self, model, gen, steps, char_table):
        self._model = model
        self._gen = gen
        self._steps = steps or 10
        self._char_table = char_table

    def evaluate(self):
        scores = []

        adapter = self._model.get_adapter()
        for i, example in enumerate(self._gen):
            if i > self._steps:
                break

            image_path, ground_true_text = example
            image = tf.keras.preprocessing.image.load_img(
                image_path, color_mode="grayscale"
            )

            expected_labels = [
                [self._char_table.get_label(ch) for ch in ground_true_text]
            ]
            inputs = adapter.adapt_x(image)

            predictions = self._model.predict(inputs)
            cer = compute_cer(expected_labels, predictions.tolist())[0]
            scores.append(cer)

        return np.array(scores).mean()


def get_bin_end(signal, s):
    for i in range(s, len(signal)):
        if not signal[i]:
            return i

    return len(signal) - 1


def get_next_bin(signal, s, min_size=30):
    for i in range(s, len(signal)):
        if signal[i]:
            end_point = get_bin_end(signal, i + 1)
            if end_point - i > min_size:
                return i, end_point

    return None


def line_segmentation(a, threshold=5):
    densities = np.mean(a, axis=1)

    binary_signal = densities > threshold

    lines = []
    last_end = 0
    while True:
        ivl = get_next_bin(binary_signal, last_end)
        if ivl is None:
            break

        i, j = ivl

        delta = 15
        slice_from = max(0, i - delta)
        slice_to = min(j + delta, len(a))
        lines.append(a[slice_from:slice_to, :])

        last_end = j + 1

    return lines


def recognize_line(model, image_path, image_height, char_table):
    x = prepare_x(image_path, image_height)
    lstm_input_shape = compute_output_shape(x.shape)
    width, channels = lstm_input_shape

    X = np.stack([x])
    input_lengths = np.array(width).reshape(1, 1)
    labellings = predict_labels(model, X, input_lengths)

    labels = labellings[0]
    s = "".join([char_table.get_character(code) for code in labels])
    return s


def recognize_document(model, image_path, image_height, char_table):

    img = tf.keras.preprocessing.image.load_img(image_path)
    a = tf.keras.preprocessing.image.img_to_array(img)
    a = binarize(a)
    line_images = line_segmentation(a)

    lines = []
    for image_array in line_images:

        image = tf.keras.preprocessing.image.array_to_img(255.0 - image_array)
        image.save("tmp.png")
        s = recognize_line(model, "tmp.png", image_height, char_table)
        lines.append(s)

    return "\n".join(lines)

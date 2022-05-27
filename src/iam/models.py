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

from utilsiam import *
from datautils import *


class HTRModel:
    def get_preprocessor(self):
        raise NotImplementedError

    def get_adapter(self):
        raise NotImplementedError

    def fit(
        self,
        train_generator,
        val_generator,
        compilation_params=None,
        training_params=None,
        **kwargs
    ):
        raise NotImplementedError

    def predict(self, inputs, **kwargs):
        raise NotImplementedError

    def save(self, path, preprocessing_params):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError

    def save_model_params(
        self, params_path, model_class_name, model_params, preprocessing_params
    ):
        d = {
            "model_class_name": model_class_name,
            "params": model_params,
            "preprocessing": preprocessing_params,
        }

        s = json.dumps(d)
        with open(params_path, "w") as f:
            f.write(s)

    @staticmethod
    def create(model_path):
        return CtcModel.load(model_path)

def compute_output_shape(input_shape):
    height, width, channels = input_shape
    new_width = width // 2 // 2 // 2
    return new_width, 80


def create_conv_model(channels=3):
    def concat(X):
        t = tf.keras.layers.Concatenate(axis=1)(tf.unstack(X, axis=3))
        return tf.transpose(t, [0, 2, 1])

    column_wise_concat = tf.keras.layers.Lambda(concat)
    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Conv2D(
            input_shape=(None, None, channels),
            filters=16,
            kernel_size=(3, 3),
            padding="same",
            activation=None,
        )
    )
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), padding="same", activation=None
        )
    )
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(
        tf.keras.layers.Conv2D(
            filters=48, kernel_size=(3, 3), padding="same", activation=None
        )
    )
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding="same", activation=None
        )
    )
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(
        tf.keras.layers.Conv2D(
            filters=80, kernel_size=(3, 3), padding="same", activation=None
        )
    )
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(column_wise_concat)

    return model


class CtcModel(HTRModel):
    def __init__(self, units, num_labels, height, channels=3):
        self._units = units
        self._num_labels = num_labels
        self._height = height
        self._channels = channels

        inp = tf.keras.layers.Input(shape=(height, None, channels))

        lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units, return_sequences=True)
        )
        densor = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=num_labels + 1, activation="softmax")
        )

        x = inp
        convnet = create_conv_model(channels)
        x = convnet(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = lstm(x)

        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units, return_sequences=True)
        )(x)

        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units, return_sequences=True)
        )(x)

        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units, return_sequences=True)
        )(x)

        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units, return_sequences=True)
        )(x)

        x = tf.keras.layers.Dropout(rate=0.5)(x)
        self.y_pred = densor(x)

        self.graph_input = inp

        self._weights_model = tf.keras.Model(self.graph_input, self.y_pred)
        self._preprocessing_options = {}

    def _create_training_model(self):
        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args

            return tf.keras.backend.ctc_batch_cost(
                labels, y_pred, input_length, label_length
            )

        labels = tf.keras.layers.Input(name="the_labels", shape=[None], dtype="float32")
        input_length = tf.keras.layers.Input(
            name="input_length", shape=[1], dtype="int64"
        )
        label_length = tf.keras.layers.Input(
            name="label_length", shape=[1], dtype="int64"
        )

        loss_out = tf.keras.layers.Lambda(
            ctc_lambda_func, output_shape=(1,), name="ctc"
        )([self.y_pred, labels, input_length, label_length])

        return tf.keras.Model(
            inputs=[self.graph_input, labels, input_length, label_length],
            outputs=loss_out,
        )

    def _create_inference_model(self):
        return tf.keras.Model(self.graph_input, self.y_pred)

    def fit(
        self,
        train_generator,
        val_generator,
        compilation_params=None,
        training_params=None,
        **kwargs
    ):
        # print(train_generator.batch_dim)
        # train_np = train_generator.numpy()
        # val_np = val_generator.numpy()
        # steps_per_epoch = math.ceil(train_generator.size / train_generator.batch_size)
        # val_steps = math.ceil(val_generator.size / val_generator.batch_size)
        # print(len(train_generator))
        steps_per_epoch = math.ceil(tf.data.experimental.cardinality(train_generator).numpy() / len(train_generator))
        val_steps = math.ceil(tf.data.experimental.cardinality(val_generator).numpy() / len(val_generator))
        print(steps_per_epoch, val_steps)

        loss = self._get_loss()
        lr = 0.001

        training_model = self._create_training_model()

        compilation_params = compilation_params or {}
        training_params = training_params or {}

        if "optimizer" in compilation_params:
            optimizer = compilation_params["optimizer"]
        else:
            optimizer = tf.keras.optimizers.Adam(lr=lr)

        if "metrics" in compilation_params:
            metrics = compilation_params["metrics"]
        else:
            metrics = []

        training_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        # print(train_generator.__iter__)

        training_model.fit(
            train_generator.__iter__(),
            # steps_per_epoch=steps_per_epoch,
            validation_data=val_generator.__iter__(),
            # validation_steps=val_steps,
            **training_params
        )

    def _get_inference_model(self):
        return self._create_inference_model()

    def get_preprocessor(self):
        preprocessor = Cnn1drnnCtcPreprocessor()
        preprocessor.configure(**self._preprocessing_options)
        return preprocessor

    def get_adapter(self):
        return CTCAdapter()

    def predict(self, inputs, **kwargs):
        X, input_lengths = inputs
        ypred = self._get_inference_model().predict(X)
        labels = decode_greedy(ypred, input_lengths)
        return labels

    def save(self, path, preprocessing_params):
        if not os.path.exists(path):
            os.mkdir(path)

        params_path = os.path.join(path, "params.json")
        weights_path = os.path.join(path, "weights.h5")

        model_params = dict(
            units=self._units,
            num_labels=self._num_labels,
            height=self._height,
            channels=self._channels,
        )
        self.save_model_params(
            params_path, "CtcModel", model_params, preprocessing_params
        )
        self._weights_model.save_weights(weights_path)

        inference_model = self._get_inference_model()

        inference_model_path = os.path.join(path, "inference_model.h5")
        inference_model.save(inference_model_path)

    @classmethod
    def load(cls, path):
        params_path = os.path.join(path, "params.json")
        weights_path = os.path.join(path, "weights.h5")
        with open(params_path) as f:
            s = f.read()

        d = json.loads(s)

        params = d["params"]
        instance = cls(**params)

        instance._weights_model.load_weights(weights_path)
        instance._preprocessing_options = d["preprocessing"]
        return instance

    def _get_loss(self):
        return {"ctc": lambda y_true, y_pred: y_pred}


def decode_greedy(inputs, input_lengths):
    with tf.compat.v1.Session() as sess:
        inputs = tf.transpose(inputs, [1, 0, 2])
        decoded, _ = tf.nn.ctc_greedy_decoder(inputs, input_lengths.flatten())

        dense = tf.sparse.to_dense(decoded[0])
        res = sess.run(dense)
        return res


def beam_search_decode(inputs, input_lengths):
    with tf.compat.v1.Session() as sess:
        inputs = tf.transpose(inputs, [1, 0, 2])
        decoded, log_probs = tf.nn.ctc_beam_search_decoder(
            inputs, input_lengths.flatten(), beam_width=10
        )
        print(log_probs)
        dense = tf.sparse.to_dense(decoded[0])
        res = sess.run(dense)
        return res


class BatchAdapter:
    def fit(self, batches):
        pass

    def adapt_x(self, image):
        raise NotImplementedError

    def adapt_batch(self, batch):
        raise NotImplementedError

    def _pad_labellings(self, labellings, target_length, padding_code=0):
        padded_labellings = []
        for labels in labellings:
            padding_size = target_length - len(labels)

            if padding_size < 0:
                # if labelling length is larger than target_length, chomp excessive characters off
                new_labelling = labels[:target_length]
            else:
                new_labelling = labels + [padding_code] * padding_size
            assert len(new_labelling) > 0
            padded_labellings.append(new_labelling)

        return padded_labellings

    def _pad_array_width(self, a, target_width):
        return pad_array_width(a, target_width)

    def _pad_image_arrays(self, image_arrays, target_width):
        return [self._pad_array_width(a, target_width) for a in image_arrays]


class CTCAdapter(BatchAdapter):
    def compute_input_lengths(self, image_arrays):
        batch_size = len(image_arrays)
        lstm_input_shapes = [compute_output_shape(a.shape) for a in image_arrays]
        widths = [width for width, channels in lstm_input_shapes]
        return np.array(widths, dtype=np.int32).reshape(batch_size, 1)

    def adapt_batch(self, batch):
        image_arrays, labellings = batch

        current_batch_size = len(labellings)

        target_width = max([a.shape[1] for a in image_arrays])
        padded_arrays = self._pad_image_arrays(image_arrays, target_width)

        X = np.array(padded_arrays).reshape(current_batch_size, *padded_arrays[0].shape)

        target_length = max([len(labels) for labels in labellings])
        padded_labellings = self._pad_labellings(labellings, target_length)

        labels = np.array(padded_labellings, dtype=np.int32).reshape(
            current_batch_size, -1
        )

        input_lengths = self.compute_input_lengths(image_arrays)

        label_lengths = np.array(
            [len(labelling) for labelling in labellings], dtype=np.int32
        ).reshape(current_batch_size, 1)

        return [X, labels, input_lengths, label_lengths], labels

    def adapt_x(self, image):
        a = tf.keras.preprocessing.image.img_to_array(image)
        x = a / 255.0

        X = np.array(x).reshape(1, *x.shape)

        input_lengths = self.compute_input_lengths(X)

        return X, input_lengths


def predict_labels(model, X, input_lengths, beam_search=False):
    densities = model.predict(X)

    if beam_search:
        predicted_labels = beam_search_decode(densities, input_lengths)
    else:
        predicted_labels = decode_greedy(densities, input_lengths)
    return predicted_labels.tolist()

def fit_model(
    model,
    train_path,
    val_path,
    char_table,
    batch_size,
    debug_interval,
    model_save_path,
    epochs,
    augment,
    lr,
):
    path = Path(train_path)

    with open(os.path.join(path.parent, "preprocessing.json")) as f:
        s = f.read()

    preprocessing_params = json.loads(s)

    adapter = model.get_adapter()

    train_generator = LinesGenerator(
        train_path, char_table, batch_size, augment=augment, batch_adapter=adapter
    )

    val_generator = LinesGenerator(
        val_path, char_table, batch_size, batch_adapter=adapter
    )

    checkpoint = MyModelCheckpoint(model, model_save_path, preprocessing_params)

    cer_generator = CompiledDataset(train_path)
    cer_val_generator = CompiledDataset(val_path)
    CER_metric = CerCallback(
        char_table,
        cer_generator,
        cer_val_generator,
        model,
        steps=5,
        interval=debug_interval,
    )

    callbacks = [checkpoint, CER_metric]

    compilation_params = dict(optimizer=tf.keras.optimizers.Adam(lr=lr))
    training_params = dict(epochs=epochs, callbacks=callbacks)
    model.fit(train_generator, val_generator, compilation_params, training_params)


def fit_ctc_model(args):
    dataset_path = args.ds
    model_save_path = args.model_path
    batch_size = args.batch_size
    units = args.units
    lr = args.lr
    epochs = args.epochs
    debug_interval = args.debug_interval
    augment = args.augment

    print("augment is {}".format(augment))

    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "validation")

    meta_info = get_meta_info(path=train_path)

    image_height = meta_info["average_height"]

    char_table_path = os.path.join(dataset_path, "character_table.txt")

    char_table = CharTable(char_table_path)

    model = CtcModel(
        units=units, num_labels=char_table.size, height=image_height, channels=1
    )
    if args.resume == True:
        print("resuming model")
        model.load("conv_lstm_model")

    fit_model(
        model,
        train_path,
        val_path,
        char_table,
        batch_size,
        debug_interval,
        model_save_path,
        epochs,
        augment,
        lr,
    )
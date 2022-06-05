import json
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import scipy
import tensorflow as tf
from PIL import Image
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

from utilsiam import *

"""
This file takes care of everything related to the preprocessing of the data, loading the data, and creating the datasets
"""


def create_lines_dataset(
    data_source,
    preprocessor,
    size=10000,
    train_fraction=0.6,
    val_fraction=0.2,
):
    """
    This function creates a character table for the dataset. It is used to map characters to indices. This is also stored so we can use it without having to generate it again.
    """
    destination_folder = "temp_ds"
    temp_folder = os.path.join(destination_folder, "extracted_lines")

    extract_lines(data_source, temp_folder, size, train_fraction, val_fraction)

    train_path = os.path.join(temp_folder, "train")
    val_path = os.path.join(temp_folder, "validation")
    test_path = os.path.join(temp_folder, "test")

    preprocessor.fit(train_path, val_path, test_path)
    preprocessor_path = os.path.join(destination_folder, "preprocessing.json")
    preprocessor.save(preprocessor_path)
    # Create preprocessed and split dataset
    split_folders = ["train", "validation", "test"]
    for folder in split_folders:
        src_dir = os.path.join(temp_folder, folder)
        dest_dir = os.path.join(destination_folder, folder)
        preprocess_images(src_dir, dest_dir, preprocessor)

    # Create a character table
    char_table_file_name = "character_table.txt"
    char_table_src = os.path.join(temp_folder, char_table_file_name)
    char_table_dest = os.path.join(destination_folder, char_table_file_name)

    shutil.copyfile(char_table_src, char_table_dest)


def preprocess_images(source, destination, preprocessor):
    """
    This function preprocesses the images in the source folder and saves them in the destination folder.
    """

    shutil.copytree(source, destination)

    ds = CompiledDataset(destination)
    for image_path, _ in ds:
        img = preprocessor.process(image_path)
        img.save(image_path)

    meta_path = os.path.join(destination, "meta.json")
    os.remove(meta_path)
    create_meta_information(destination)


def extract_lines(
    data_source,
    destination_folder="lines_dataset",
    size=10000,
    train_fraction=0.6,
    val_fraction=0.2,
):
    """
    This function extracts the lines from the data source and saves them in the destination folder. Mostly useful to make the character table. In theory should be fused with line segmentation but that was already done for the dataset
    """
    dest_to_copier = {}
    dest_texts = {}

    num_created = 0

    example_generator = data_source.__iter__()
    for triple in split_examples(example_generator, size, train_fraction, val_fraction):
        folder_name, file_path, text = triple
        split_destination = os.path.join(destination_folder, folder_name)
        if folder_name not in dest_to_copier:
            dest_to_copier[folder_name] = FileCopier(split_destination)

        if split_destination not in dest_texts:
            dest_texts[split_destination] = []

        copier = dest_to_copier[folder_name]

        copier.copy_file(file_path)

        dest_texts[split_destination].append(text)

        num_created += 1
        if num_created % 500 == 0:
            completed_percentage = num_created / float(size) * 100
            print(
                "Created {} out of {} lines. {} % done".format(
                    num_created, size, completed_percentage
                )
            )

    for split_folder in dest_texts.keys():
        lines_path = os.path.join(split_folder, "lines.txt")
        with open(lines_path, "w") as f:
            for line in dest_texts[split_folder]:
                f.write(line + "\n")

        print("Creating meta information for {} split folder".format(split_folder))
        create_meta_information(split_folder)

    print("Creating a character table")

    split_folders = dest_texts.keys()
    char_table_lines = create_char_table(split_folders)

    char_table_path = os.path.join(destination_folder, "character_table.txt")
    with open(char_table_path, "w") as f:
        f.write(char_table_lines)


class FileCopier:
    """
    Copies the files in the source folder to the destination folder.
    """

    def __init__(self, folder):
        self._folder = folder
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)

        self._num_copied = len(os.listdir(self._folder))

    def copy_file(self, obj):
        if type(obj) is str:
            # obj must be path to image file
            file_path = obj
            _, ext = os.path.splitext(file_path)

            dest = os.path.join(self._folder, str(self._num_copied) + ext)
            shutil.copyfile(file_path, dest)
        else:
            # obj must be Pillow image
            dest = os.path.join(self._folder, str(self._num_copied) + ".png")
            obj.save(dest)

        self._num_copied += 1
        return dest


def split_examples(example_generator, size, train_fraction=0.6, val_fraction=0.2):
    """
    This function splits the examples into train, validation and test. But is a generator function
    """
    train_folder = "train"
    val_folder = "validation"
    test_folder = "test"

    folders = [train_folder, val_folder, test_folder]

    for count, example in enumerate(example_generator):
        if count > size:
            break

        test_fraction = 1 - train_fraction - val_fraction
        pmf = [train_fraction, val_fraction, test_fraction]
        destination = np.random.choice(folders, p=pmf)

        yield (destination,) + example


def create_meta_information(dataset_path):
    """
    This function creates the meta information file for the dataset. Basically it creates a json file with all the information about the dataset.
    """
    widths = []
    heights = []

    for fname in os.listdir(dataset_path):
        _, ext = os.path.splitext(fname)
        if ext != ".txt":
            image_path = os.path.join(dataset_path, fname)
            image = tf.keras.preprocessing.image.load_img(image_path)
            widths.append(image.width)
            heights.append(image.height)

    lines_path = os.path.join(dataset_path, "lines.txt")

    text_lengths = []
    with open(lines_path) as f:
        for row in f.readlines():
            line = row.rstrip("\n")
            text_lengths.append(len(line))

    max_width = int(np.max(widths))
    max_height = int(np.max(heights))
    min_width = int(np.min(widths))
    min_height = int(np.min(heights))
    average_width = int(np.mean(widths))
    average_height = int(np.mean(heights))
    max_text_length = int(np.max(text_lengths))
    num_examples = len(widths)

    d = dict(
        max_width=max_width,
        max_height=max_height,
        min_width=min_width,
        min_height=min_height,
        average_width=average_width,
        average_height=average_height,
        max_text_length=max_text_length,
        num_examples=num_examples,
    )

    s = json.dumps(d)
    meta_path = os.path.join(dataset_path, "meta.json")
    with open(meta_path, "w") as f:
        f.write(s)


def create_char_table(split_folders):
    """
    This function creates a character table for the dataset.
    """
    chars = set()
    for folder in split_folders:
        lines_path = os.path.join(folder, "lines.txt")
        with open(lines_path) as f:
            for line in f.readlines():
                text = line.rstrip()
                line_chars = list(text)
                chars = chars.union(line_chars)

    char_table = "\n".join(list(chars))
    return char_table


def get_image_array(image_path, target_height):
    """
    This function returns the image array for the given image path.
    """
    img = tf.keras.preprocessing.image.load_img(image_path)

    aspect_ratio = img.width / img.height

    new_width = int(target_height * aspect_ratio)
    if new_width <= target_height:
        return pad_image(img, target_height, target_height + 1)

    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(target_height, new_width)
    )

    return tf.keras.preprocessing.image.img_to_array(img)


def pad_array_width(a, target_width):
    """
    This function pads the array with zeros to the target width.
    """
    width = a.shape[1]

    right_padding = target_width - width

    if right_padding < 0:
        # if image width is larger than target_width, crop the image
        return a[:, :target_width]

    horizontal_padding = (0, right_padding)
    vertical_padding = (0, 0)
    depth_padding = (0, 0)
    return scipy.pad(a, pad_width=[vertical_padding, horizontal_padding, depth_padding])


def pad_image(img, target_height, target_width):
    """
    This function pads the image with zeros to the target height and width.
    """
    a = tf.keras.preprocessing.image.img_to_array(img)

    _, _, original_channels = a.shape

    im = np.ones((target_height, target_width, original_channels),
                 dtype=np.float) * 255

    cropped = a[:target_height, :target_width]
    for i in range(cropped.shape[0]):
        for j in range(cropped.shape[1]):
            im[i, j, :] = a[i, j, :]

    return im


def binarize(image_array, threshold=200, invert=True):
    """
    This function binarizes the image array. Probably not too useful for this project as the images are already binarized.
    """
    h, w, channels = image_array.shape
    grayscale = rgb_to_grayscale(image_array)
    black_mask = grayscale < threshold
    white_mask = grayscale >= threshold

    if invert:
        tmp = white_mask
        white_mask = black_mask
        black_mask = tmp

    grayscale[white_mask] = 255
    grayscale[black_mask] = 0

    return grayscale.reshape((h, w, 1))


def rgb_to_grayscale(a):
    """
    This function converts the image array to grayscale.
    """
    return a[:, :, 0] * 0.2125 + a[:, :, 1] * 0.7154 + a[:, :, 2] * 0.0721


class Cnn1drnnCtcPreprocessor():
    """
    This class preprocesses the images and labels for the CNN-1D-RNN-CTC model.
    """
    def __init__(self):
        self._average_height = 50

    def configure(self, average_height=50):
        self._average_height = average_height

    def fit(self, train_path, val_path, test_path):
        train_ds = CompiledDataset(train_path)
        self._average_height = train_ds.average_height

    def process(self, image_path):
        image_array = get_image_array(image_path, self._average_height)
        a = binarize(image_array)
        return tf.keras.preprocessing.image.array_to_img(a)

    def save(self, path):
        d = {"average_height": self._average_height}
        self._save_dict(path, d)

    def _save_dict(self, path, d):
        s = json.dumps(d)
        with open(path, "w") as f:
            f.write(s)


class CompiledDataset:
    """
    This class represents a compiled dataset which just has all the data in one place along with some metadata. Also it has a dataset iterator.
    """
    def __init__(self, dataset_root):
        self._root = dataset_root
        self._lines = []

        lines_path = os.path.join(dataset_root, "lines.txt")
        with open(lines_path) as f:
            for row in f.readlines():
                self._lines.append(row.rstrip("\n"))

        meta_path = os.path.join(dataset_root, "meta.json")
        with open(meta_path) as f:
            s = f.read()
        meta_info = json.loads(s)

        self.__dict__.update(meta_info)

        self._num_examples = meta_info["num_examples"]

        os.path.dirname(dataset_root)

    @property
    def size(self):
        return self._num_examples

    def __iter__(self):
        for i in range(self._num_examples):
            yield self.get_example(i)

    def get_example(self, line_index):
        text = self._lines[line_index]
        image_path = os.path.join(self._root, str(line_index) + ".png")
        return image_path, text


class LinesGenerator():
    def __init__(
        self, dataset_root, char_table, batch_size=4, augment=False, batch_adapter=None
    ):
        """
        This class generates batches of lines for the CNN-1D-RNN-CTC model.
        """
        self._root = dataset_root
        self._char_table = char_table
        self._batch_size = batch_size
        self._augment = augment

        self._adapter = batch_adapter

        self._ds = CompiledDataset(dataset_root)

        self._indices = list(range(self._ds.size))

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def size(self):
        return self._ds.size

    def __iter__(self):
        batches_gen = self.get_batches()
        self._adapter.fit(batches_gen)

        while True:
            for batch in self.get_batches():
                yield self._adapter.adapt_batch(batch)

    def get_batches(self):
        random.shuffle(self._indices)
        image_arrays = []
        labellings = []
        for line_index in self._indices:
            image_array, labels = self.get_example(line_index)
            image_arrays.append(image_array)
            labellings.append(labels)

            if len(labellings) >= self._batch_size:
                batch = image_arrays, labellings
                image_arrays = []
                labellings = []
                yield batch

        if len(labellings) >= 1:
            yield image_arrays, labellings

    def text_to_class_labels(self, text):
        """
        This function converts the text to class labels for a line.
        """
        return [self._char_table.get_label(ch) for ch in text]

    def get_example(self, line_index):
        """
        This function returns the image array and the class labels for a line.
        """
        image_path, text = self._ds.get_example(line_index)
        img = tf.keras.preprocessing.image.load_img(
            image_path, color_mode="grayscale")
        a = tf.keras.preprocessing.image.img_to_array(img)
        x = a / 255.0
        y = self.text_to_class_labels(text)
        return x, y


class CharTable:
    """
    This class represents a character table.
    """
    def __init__(self, char_table_path):
        self._char_to_label, self._label_to_char = self.load_char_table(
            char_table_path)

        self._max_label = max(self._label_to_char.keys())  # This is the maximum number of labels.

    def load_char_table(self, path):
        """
        This function loads the character table.
        """
        char_to_label = {}
        label_to_char = {}
        with open(path) as f:
            for label, line in enumerate(f.readlines()):
                ch = line.rstrip("\n")
                char_to_label[ch] = label
                label_to_char[label] = ch

        return char_to_label, label_to_char

    @property
    def size(self):
        return len(self._char_to_label) + 2

    @property
    def sos(self): #start of string
        return self._max_label + 1

    @property
    def eos(self): #end of string
        return self.sos + 1

    def get_label(self, ch):
        return self._char_to_label[ch]

    def get_character(self, class_label):
        """
        This function returns the character for a class label.
        """
        if class_label == self.sos:
            return ""

        if class_label == self.eos:
            return "\n"

        return self._label_to_char[class_label]


class IAM:
    """
    Class for reading the IAM dataset
    """

    def __init__(
        self,
        path="/media/hdd/github/handwriting_recog_codes/data/IAM-data/",
        subset=None,
    ):
        self.main_path = Path(path)
        self.images_path = self.main_path / "img"
        self.labels_path = self.main_path / "iam_lines_gt.txt"
        self.batch_size = 64
        self.padding_token = 99
        self.subset = subset
        self.images, self.labels = self.iam_data_reader(
            self.images_path, self.labels_path, self.subset
        )

    def __iter__(self):
        for image, label in zip(self.images, self.labels):
            yield Image.open(image), label

    def iam_data_reader(self, images_path, labels_path, subset=None):
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000)

    args = parser.parse_args()
    size = args.size
    destination = "temp_ds"

    if not os.path.isdir(destination):
        os.makedirs(destination)

    shutil.rmtree(destination)
    create_lines_dataset(
        IAM(),
        Cnn1drnnCtcPreprocessor(),
        size=size,
        train_fraction=0.8,
        val_fraction=0.1,
    )
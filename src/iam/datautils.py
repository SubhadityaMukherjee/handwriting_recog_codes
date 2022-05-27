import json
import logging
import math
import os
import random
import shutil
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


def extract_texts_with_ids(path):
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ParseError:
        raise Exception()

    root_tag = list(root.iterfind("handwritten-part"))
    assert len(root_tag) != 0

    for line in root_tag[0].iterfind("line"):
        for word in line.iterfind("word"):
            assert "text" in word.attrib and "id" in word.attrib
            text = word.attrib["text"]
            file_id = word.attrib["id"]
            yield text, file_id


def build_words_dataset(
    words_root="iam_database/iam_words",
    xml_root="iam_database/iam_database_xml",
    destination_folder="words_dataset",
    size=10000,
    train_fraction=0.6,
    val_fraction=0.2,
):
    if os.path.exists(destination_folder):
        raise Exception("Data set already exists!")

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    label_maker = LabelMaker()

    count = 0

    train_folder = os.path.join(destination_folder, "train")
    val_folder = os.path.join(destination_folder, "validation")
    test_folder = os.path.join(destination_folder, "test")
    train_dataset_creator = DataSetCreator(
        words_root, train_folder, label_maker)
    val_dataset_creator = DataSetCreator(words_root, val_folder, label_maker)
    test_dataset_creator = DataSetCreator(words_root, test_folder, label_maker)

    creators = [train_dataset_creator,
                val_dataset_creator, test_dataset_creator]

    for word, file_id in get_words_with_file_ids(xml_root):
        if count > size:
            break
        print(word, file_id, count, size)
        test_fraction = 1 - train_fraction - val_fraction
        pmf = [train_fraction, val_fraction, test_fraction]
        dataset_creator = np.random.choice(creators, p=pmf)

        dataset_creator.add_example(word, file_id)
        count += 1

    for dataset_creator in creators:
        dataset_creator.create_paths_file()

    dictionary_file = os.path.join(destination_folder, "dictionary.txt")

    with open(dictionary_file, "w") as f:
        for word in label_maker.words:
            f.write(word + "\n")


def get_words_with_file_ids(xml_root):
    for xml_path in file_iterator(xml_root):
        for word, file_id in extract_texts_with_ids(xml_path):
            if word.isalnum():
                yield word, file_id


class DataSetCreator:
    def __init__(self, words_root, destination, label_maker):
        if not os.path.exists(destination):
            os.makedirs(destination)

        self._finder = PathFinder(words_root)
        self._label_to_copier = {}
        self._label_maker = label_maker
        self._destination_folder = destination
        self._dataset_paths = []

    def add_example(self, word, file_id):
        file_path = self._finder.find_path(file_id)

        self._label_maker.make_label_if_not_exists(word)
        label = self._label_maker[word]
        label_string = str(label)

        if label_string not in self._label_to_copier:
            folder_path = os.path.join(self._destination_folder, label_string)
            self._label_to_copier[label_string] = FileCopier(folder_path)

        copier = self._label_to_copier[label_string]
        copy_path = copier.copy_file(file_path)

        self._dataset_paths.append(copy_path)

    def create_paths_file(self):
        paths_file = os.path.join(self._destination_folder, "paths_list.txt")

        with open(paths_file, "w") as f:
            for path in self._dataset_paths:
                f.write(path + "\n")


def create_lines_dataset(
    data_source,
    preprocessor,
    size=10000,
    train_fraction=0.6,
    val_fraction=0.2,
):
    destination_folder = "temp_ds"
    temp_folder = os.path.join(destination_folder, "extracted_lines")

    extract_lines(data_source, temp_folder, size, train_fraction, val_fraction)

    train_path = os.path.join(temp_folder, "train")
    val_path = os.path.join(temp_folder, "validation")
    test_path = os.path.join(temp_folder, "test")

    preprocessor.fit(train_path, val_path, test_path)
    preprocessor_path = os.path.join(destination_folder, "preprocessing.json")
    preprocessor.save(preprocessor_path)

    split_folders = ["train", "validation", "test"]

    for folder in split_folders:
        src_dir = os.path.join(temp_folder, folder)
        dest_dir = os.path.join(destination_folder, folder)
        preprocess_images(src_dir, dest_dir, preprocessor)

    char_table_file_name = "character_table.txt"
    char_table_src = os.path.join(temp_folder, char_table_file_name)
    char_table_dest = os.path.join(destination_folder, char_table_file_name)

    shutil.copyfile(char_table_src, char_table_dest)


def preprocess_images(source, destination, preprocessor):

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


class LabelMaker:
    def __init__(self):
        self._word_to_label = {}
        self._label_to_word = []

    @property
    def num_labels(self):
        return len(self._word_to_label)

    @property
    def words(self):
        return self._label_to_word

    def make_label_if_not_exists(self, word):
        if word not in self._word_to_label:
            self._label_to_word.append(word)
            self._word_to_label[word] = self.num_labels

    def __getitem__(self, word):
        return self._word_to_label[word]


class ConnectedComponent:
    def __init__(self, points):
        self.points = points

        self.y = [y for y, x in points]
        self.x = [x for y, x in points]
        self.top = min(self.y)
        self.bottom = max(self.y)

        self.left = min(self.x)
        self.right = max(self.x)

        self.height = self.bottom - self.top + 1
        self.width = self.right - self.left + 1

        self.center_x = np.array(self.x).mean()
        self.center_y = self.top + self.height // 2

    @property
    def bounding_box(self):
        return self.left, self.bottom, self.right, self.top

    def __contains__(self, point):
        y, x = point
        return x >= self.left and x <= self.right and y >= self.top and y <= self.bottom

    def visualize(self):
        a = np.zeros((self.bottom + 1, self.right + 1, 1))

        for y, x in self.points:
            a[y, x, 0] = 255

        tf.keras.preprocessing.image.array_to_img(a).show()


class Line:
    def __init__(self):
        self._components = []

    def add_component(self, component):
        self._components.append(component)

    def __iter__(self):
        for c in self._components:
            yield c

    @property
    def num_components(self):
        return len(self._components)

    @property
    def top(self):
        return min([component.top for component in self._components])

    @property
    def bottom(self):
        return max([component.bottom for component in self._components])

    @property
    def left(self):
        return min([component.left for component in self._components])

    @property
    def right(self):
        return max([component.right for component in self._components])

    @property
    def height(self):
        return self.bottom - self.height

    def __contains__(self, component):
        padding = 5
        return (
            component.center_y >= self.top - padding
            and component.center_y < self.bottom + padding
        )


def to_vertex(i, j, w):
    return i * w + j


def to_grid_cell(v, h, w):
    row = v // w
    col = v % w
    return row, col


def is_within_bounds(h, w, i, j):
    return i < h and i >= 0 and j < w and j >= 0


def make_edges(h, w, i, j):
    if j >= w:
        return []

    x = j
    y = i

    neighbors = []
    for l in [-1, 1]:
        for m in [-1, 1]:
            neighbors.append((y + l, x + m))

    vertices = [
        to_vertex(y, x, w) for y, x in neighbors if is_within_bounds(h, w, y, x)
    ]

    u = to_vertex(i, j, w)
    edges = [(u, v) for v in vertices]
    return edges


def make_grid_graph(im):
    h, w = im.shape

    G = nx.Graph()

    for i in range(h):
        for j in range(w):
            for u, v in make_edges(h, w, i, j):
                row, col = to_grid_cell(v, h, w)
                if im[i, j] > 0 and im[row, col] > 0:
                    G.add_node(to_vertex(i, j, w))
                    G.add_node(u)
                    G.add_edge(u, v)

    return G


def get_connected_components(im):
    G = make_grid_graph(im)

    h, w = im.shape

    components = []
    for vertices in nx.connected_components(G):
        points = []
        for v in vertices:
            point = to_grid_cell(v, h, w)
            points.append(point)

        if len(points) > 0:
            components.append(ConnectedComponent(points))

    return components


def get_seam(signed_distance):
    s = ""
    h, w, _ = signed_distance.shape

    signed_distance = signed_distance.reshape(h, w)
    for row in signed_distance.tolist():
        s += " ".join(map(str, row)) + "\n"

    with open("array.txt", "w") as f:
        f.write("{} {}\n".format(h, w))
        f.write(s)

    subprocess.call(["./carving"])

    with open("seam.txt") as f:
        s = f.read()

    row_indices = [int(v) for v in s.split(" ") if v != ""]

    column_indices = list(range(w))

    return row_indices, column_indices


def visualize_map(m):
    h, w, _ = m.shape
    m = m.reshape(h, w)
    m = m + abs(m.min())

    c = m.max() / 255.0
    m = m / c

    m = 255 - m

    tf.keras.preprocessing.image.array_to_img(m.reshape(h, w, 1)).show()


def visualize_components(line):
    h = line.bottom + 1
    w = line.right + 1
    a = np.zeros((h, w, 1))
    for comp in line:
        for y, x in comp.points:
            a[y, x] = 255

    tf.keras.preprocessing.image.array_to_img(a.reshape(h, w, 1)).show()


def prepare_image():
    # img = tf.keras.preprocessing.image.load_img('iam_database/iam_database_formsA-D/a01-000u.png')
    img = tf.keras.preprocessing.image.load_img("screen.png")

    a = tf.keras.preprocessing.image.img_to_array(img)
    h, w, _ = a.shape

    a = binarize(a)
    x = a.reshape(h, w)

    return x // 255


def get_intersections(components, seam, lines):
    row_indices, column_indices = seam

    new_line = Line()

    for row, col in zip(row_indices, column_indices):
        point = (row, col)
        for component in components[:]:
            if point in component:
                add_to_new_line = True

                for line in lines:
                    if component in line:
                        line.add_component(component)
                        add_to_new_line = False
                        break
                if add_to_new_line:
                    new_line.add_component(component)

                components.remove(component)

    if new_line.num_components > 0:
        lines.append(new_line)


def seam_carving_segmentation():

    x = prepare_image()

    x_copy = x.copy()
    h, w = x.shape

    components = get_connected_components(x)

    lines = []
    xc = 1 - x
    signed_distance = ndimage.distance_transform_edt(
        xc
    ) - ndimage.distance_transform_edt(x)
    signed_distance = signed_distance.reshape(h, w, 1)

    for i in range(h):
        if len(components) == 0:
            break

        seam = get_seam(signed_distance)
        row_indices, column_indices = seam
        signed_distance[row_indices, column_indices] = 255
        get_intersections(components, seam, lines)

        print("i", i, "lines #:", len(lines),
              "num components", len(components))

    for line in lines:
        visualize_components(line)
        input("press key\n")

    # todo: store components and line regions in R-trees
    # todo: compute all H seams in c++
    # todo: fast graph processing for large images


def get_zero_padded_array(image_path, target_height):
    img = tf.keras.preprocessing.image.load_img(image_path)
    a = pad_image_height(img, target_height)

    new_height, new_width = a.shape

    if new_width <= target_height:
        img = tf.keras.preprocessing.image.array_to_img(a)
        return pad_image(img, target_height, target_height + 1)

    return a


def get_image_array(image_path, target_height):
    img = tf.keras.preprocessing.image.load_img(image_path)

    aspect_ratio = img.width / img.height

    new_width = int(target_height * aspect_ratio)
    if new_width <= target_height:
        return pad_image(img, target_height, target_height + 1)

    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(target_height, new_width)
    )

    return tf.keras.preprocessing.image.img_to_array(img)


def pad_image_height(img, target_height):
    a = tf.keras.preprocessing.image.img_to_array(img)

    height = a.shape[0]

    padding_amount = target_height - height

    assert padding_amount >= 0

    top_padding = padding_amount // 2
    if padding_amount % 2 == 0:
        vertical_padding = (top_padding, top_padding)
    else:
        vertical_padding = (top_padding, top_padding + 1)

    horizontal_padding = (0, 0)
    depth_padding = (0, 0)
    return scipy.pad(a, pad_width=[vertical_padding, horizontal_padding, depth_padding])


def pad_array_width(a, target_width):
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
    a = tf.keras.preprocessing.image.img_to_array(img)

    original_height, original_width, original_channels = a.shape

    im = np.ones((target_height, target_width, original_channels),
                 dtype=np.float) * 255

    cropped = a[:target_height, :target_width]
    for i in range(cropped.shape[0]):
        for j in range(cropped.shape[1]):
            im[i, j, :] = a[i, j, :]

    return im


def prepare_x(image_path, image_height, should_binarize=True, transform=False):
    image_array = get_image_array(image_path, image_height)
    if should_binarize:
        a = binarize(image_array)
    else:
        a = image_array

    if transform:
        rotation_range = 1
        shift = 3
        zoom = 0.01
        image_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=shift,
            height_shift_range=shift,
            zoom_range=zoom,
        )
        gen = image_gen.flow(np.array([a]), batch_size=1)

        a = next(gen)[0]

    return a / 255.0


def binarize(image_array, threshold=200, invert=True):
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
    return a[:, :, 0] * 0.2125 + a[:, :, 1] * 0.7154 + a[:, :, 2] * 0.0721


class BasePreprocessor:
    def fit(self, train_path, val_path, test_path):
        pass

    def configure(self, **kwargs):
        pass

    def process(self, image_path):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def _save_dict(self, path, d):
        s = json.dumps(d)
        with open(path, "w") as f:
            f.write(s)


class Cnn1drnnCtcPreprocessor(BasePreprocessor):
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


class BaseGenerator:
    def __iter__(self):
        raise NotImplementedError


def get_dictionary():
    dictionary = []
    with open("words_dataset/dictionary.txt") as f:
        for i, line in enumerate(f.readlines()):
            dictionary.append(line.rstrip())
    return dictionary


class CompiledDataset:
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


class LinesGenerator(BaseGenerator):
    def __init__(
        self, dataset_root, char_table, batch_size=4, augment=False, batch_adapter=None
    ):
        self._root = dataset_root
        self._char_table = char_table
        self._batch_size = batch_size
        self._augment = augment

        if batch_adapter is None:
            self._adapter = CTCAdapter()
        else:
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
        return [self._char_table.get_label(ch) for ch in text]

    def get_example(self, line_index):
        image_path, text = self._ds.get_example(line_index)
        img = tf.keras.preprocessing.image.load_img(
            image_path, color_mode="grayscale")
        a = tf.keras.preprocessing.image.img_to_array(img)
        x = a / 255.0
        y = self.text_to_class_labels(text)
        return x, y


class CharTable:
    def __init__(self, char_table_path):
        self._char_to_label, self._label_to_char = self.load_char_table(
            char_table_path)

        self._max_label = max(self._label_to_char.keys())

    def load_char_table(self, path):
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
    def sos(self):
        return self._max_label + 1

    @property
    def eos(self):
        return self.sos + 1

    def get_label(self, ch):
        return self._char_to_label[ch]

    def get_character(self, class_label):
        if class_label == self.sos:
            return ""

        if class_label == self.eos:
            return "\n"

        return self._label_to_char[class_label]

class IAM():
    """
    Class for reading the IAM dataset
    """
    def __init__(self, path="/media/hdd/github/handwriting_recog_codes/data/IAM-data/", subset=None):
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

    response = input(
        "All existing data in the directory {} "
        "will be erased. Continue (Y/N) ?".format(destination)
    )
    if response == "Y":
        shutil.rmtree(destination)
        create_lines_dataset(
            IAM(),
            Cnn1drnnCtcPreprocessor(),
            size=size,
            train_fraction=0.8,
            val_fraction=0.1,
        )
    else:
        pass

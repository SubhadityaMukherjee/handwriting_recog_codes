import math
import os
import pathlib
from glob import glob
from tqdm import tqdm
import pandas as pd

import concurrent.futures
import matplotlib.pyplot as plt
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
from IPython.display import Image

from datautils import *
from models import *
from utilsiam import *
from spellcheck import SpellCheck

# from keras.utils.vis_utils import plot_model

"""
This module contains evaluation functions for the IAM dataset.
"""

def gridder(images, labels, col=2):
    """
    Gets a list of images and labels and displays them in a grid
    """
    image_count = len(images)
    row = math.ceil(image_count / col)
    plt.figure(figsize=(col * 4, row * 4))
    plt.figure(figsize=(col * 4, row * 4))

    for i, img_path in enumerate(images):
        img = plt.imread(img_path)
        plt.subplot(row, col, i + 1)
        if i % 2 == 0:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.title(labels[i])
        plt.axis("off")
    plt.savefig("results/results.pdf")
    plt.close()


def single_prediction(image_path, model, char_table, adapter):
    """
    Runs the model on a single image and displays the results.
    """
    try:
        image = tf.keras.preprocessing.image.load_img(
            image_path, color_mode="grayscale"
        )
        inputs = adapter.adapt_x(image)
        predictions = model.predict(inputs)
        predicted_text = codes_to_string(predictions[0], char_table)
        return predicted_text
    except:
        return ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--imagepath", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    model_path = "conv_lstm_model"
    if args.evaluate:
        # Evaluate the model on the test set for both spellcheck and not spellcheck
        folder = "temp_ds/test"
        char_table_path = "char_table_single_im.txt"
        char_table = CharTable(char_table_path)

        model = CtcModel.load(model_path)

        adapter = model.get_adapter()
        files = os.listdir(folder)
        files = [Path(folder) / f for f in files if f.endswith(".png")]
        files.sort()
        # files = files[:21]

        with open("temp_ds/test/lines.txt") as f:
            lines = f.readlines()

        sp = SpellCheck(Path("../../data/IAM-data/iam_lines_gt.txt"))
        res = [
            single_prediction(image_path, model, char_table, adapter)
            for image_path in tqdm(files)
        ]

        print("CER without spellcheck: {}".format(float(compute_cer(lines, res))))
        print("Running spellcheck")
        sp_res = [sp.correct(x) for x in tqdm(res, total=len(res))]
        print("CER with spellcheck: {}".format(float(compute_cer(lines, sp_res)[0])))
        gridder(files[:20], sp_res[:20])

    elif args.folder is not None:
        """
        Evaluate the model on a given folder for both spellcheck and not spellcheck
        """
        folder = args.folder
        char_table_path = "char_table_single_im.txt"
        char_table = CharTable(char_table_path)

        # model = HTRModel.create(model_path)
        with tf.device("/cpu:0"):
            model = CtcModel.load(model_path)

        adapter = model.get_adapter()
        dict_outs = {}
        files = os.listdir(folder)
        files = [f for f in files if f.endswith(".png")]
        l = len(files)
        sp = SpellCheck("../../data/IAM-data/iam_lines_gt.txt")

        def process_file(file):
            image_path = os.path.join(folder, file)
            predicted_text = single_prediction(image_path, model, char_table, adapter)
            predicted_text = sp.correct(predicted_text)
            with open(
                f"results/iam/iam_predictions/{file.split('.')[0]}.txt", "w+"
            ) as fle:
                fle.write(predicted_text)
            return predicted_text

        results = {fns: process_file(fns) for fns in tqdm(files)}
        print(results)
    else:
        """
        Evaluate the model on a single image
        """
        dataset_path = args.imagepath
        char_table_path = "char_table_single_im.txt"
        char_table = CharTable(char_table_path)

        model = HTRModel.create(model_path)

        adapter = model.get_adapter()
        pred = single_prediction(args.imagepath, model, char_table, adapter)

        print('Predicted: "{}"'.format(pred))

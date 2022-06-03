import concurrent.futures
import math
import os
import pathlib
from glob import glob

import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import Image
from tqdm import tqdm

from datautils import *
from models import *
from spellcheck import SpellCheck
from utilsiam import *

# from keras.utils.vis_utils import plot_model


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


def run_demo(model, gen, char_table, adapter, grid=False, spellcheck=False):
    """
    Runs the model on a dataset and displays the results for the first 20 samples in results/results.pdf . The rest are just printed.
    """
    cer_tot = 0
    cnt = 0
    ims, lbls = [], []
    sp = SpellCheck("../../data/IAM-data/iam_lines_gt.txt")
    for image_path, ground_true_text in gen.__iter__():
        image = tf.keras.preprocessing.image.load_img(
            image_path, color_mode="grayscale"
        )
        expected_labels = [[char_table.get_label(ch) for ch in ground_true_text]]

        inputs = adapter.adapt_x(image)

        predictions = model.predict(inputs)
        cer = compute_cer(expected_labels, predictions.tolist())[0]

        predicted_text = codes_to_string(predictions[0], char_table)
        if spellcheck == True:
            predicted_text = sp.correct(predicted_text)
        ims.append(image_path)
        lbls.append(predicted_text)

        print('LER {}, "{}" -> "{}"'.format(cer, ground_true_text, predicted_text))
        cer_tot += cer
        cnt += 1
        if grid == True:
            if cnt == 20:
                break
    if grid == True:
        gridder(ims[:20], lbls[:20])
    return cer_tot / cnt


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


def multiple_prediction(image_path, model, char_table, adapter):
    """
    Runs the model on a single image and displays the results.
    """
    try:
        # inputs = [adapter.adapt_x(tf.keras.preprocessing.image.load_img(x, color_mode="grayscale")) for x in image_path]
        inputs = adapter.adapt_batch(image_path)
        predictions = model.predict_multi(inputs)
        predicted_text = [codes_to_string(x[0], char_table) for x in predictions]
        return predicted_text
    except Exception as e:
        print(e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetdemo", action="store_true")
    parser.add_argument("--imagepath", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    model_path = "conv_lstm_model"

    if args.evaluate:
        folder = "temp_ds/test/"
        char_table_path = "char_table_single_im.txt"
        char_table = CharTable(char_table_path)

        model = CtcModel.load(model_path)

        adapter = model.get_adapter()
        files = os.listdir(folder)
        files = [os.path.join(folder, f) for f in files if f.endswith(".png")]
        files.sort()

        with open("temp_ds/test/lines.txt") as f:
            lines = f.readlines()
        sp = SpellCheck("../../data/IAM-data/iam_lines_gt.txt")
        res = multiple_prediction(files, model, char_table, adapter)
        sp_res = [sp.correct(x) for x in res]
        print("CER without spellcheck: {}".format(float(compute_cer(lines, res)[0])))
        print("CER with spellcheck: {}".format(float(compute_cer(lines, sp_res)[0])))

    if args.folder is not None:
        folder = args.folder
        char_table_path = "char_table_single_im.txt"
        char_table = CharTable(char_table_path)

        model = CtcModel.load(model_path)

        adapter = model.get_adapter()
        files_names = os.listdir(folder)
        # files_names = files_names[:10]

        files = [os.path.join(folder, f) for f in files_names if f.endswith(".png")]

        shutil.rmtree("results/iam_predictions", ignore_errors=True)
        if not os.path.exists("results/iam_predictions"):
            os.makedirs("results/iam_predictions")

        sp = SpellCheck("../../data/IAM-data/iam_lines_gt.txt")
        res = multiple_prediction(files, model, char_table, adapter)
        sp_res = [sp.correct(x) for x in res]
        print("Saving results")
        for i, f in tqdm(enumerate(files_names), total=len(files)):
            with open(f"results/iam_predictions/{f.split('.')[0]}.txt", "w") as fle:
                fle.write(f"{sp_res[i]}")

    if args.imagepath is not None:
        sp = SpellCheck("../../data/IAM-data/iam_lines_gt.txt")
        dataset_path = args.imagepath
        char_table_path = "char_table_single_im.txt"
        char_table = CharTable(char_table_path)

        model = CtcModel.load(model_path)

        adapter = model.get_adapter()
        pred = single_prediction(args.imagepath, model, char_table, adapter)

        print('Predicted: "{}"'.format(pred))
        print('Corrected: "{}"'.format(sp.correct(pred)))

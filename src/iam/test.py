import math
import os
import pathlib
from glob import glob
from tqdm import tqdm
import pandas as pd

import concurrent.futures
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import Image

from datautils import *
from models import *
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


def run_demo(model, gen, char_table, adapter):
    """
    Runs the model on a dataset and displays the results for the first 20 samples in results/results.pdf . The rest are just printed.
    """
    cer_tot = 0
    cnt = 0
    ims, lbls = [], []
    for image_path, ground_true_text in gen.__iter__():
        image = tf.keras.preprocessing.image.load_img(
            image_path, color_mode="grayscale"
        )
        expected_labels = [
            [char_table.get_label(ch) for ch in ground_true_text]]

        inputs = adapter.adapt_x(image)

        predictions = model.predict(inputs)
        cer = compute_cer(expected_labels, predictions.tolist())[0]

        predicted_text = codes_to_string(predictions[0], char_table)
        ims.append(image_path)
        lbls.append(predicted_text)

        print('LER {}, "{}" -> "{}"'.format(cer, ground_true_text, predicted_text))
        cer_tot += cer
        cnt += 1
        if cnt == 20:
            break
    gridder(ims[:20], lbls[:20])
    return cer_tot / cnt


def single_prediction(image_path, model, char_table, adapter):
    """
    Runs the model on a single image and displays the results.
    """
    try:
        image = tf.keras.preprocessing.image.load_img(
            image_path, color_mode="grayscale")
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
    args = parser.parse_args()

    model_path = "conv_lstm_model"
    if args.dataset is not None:
        dataset_path = args.dataset
        char_table_path = os.path.join(
            os.path.dirname(dataset_path), "character_table.txt"
        )
        char_table = CharTable(char_table_path)

        model = HTRModel.create(model_path)
        # plot_model(model._weights_model, to_file='model_plot.png', show_shapes=True)

        ds = CompiledDataset(dataset_path)

        adapter = model.get_adapter()
        cer_score = run_demo(model, ds, char_table, adapter=adapter)
        print(f"Avg cer {cer_score}")

    elif args.folder is not None:
        folder = args.folder
        char_table_path = "char_table_single_im.txt"
        char_table = CharTable(char_table_path)

        model = HTRModel.create(model_path)

        adapter = model.get_adapter()
        files = os.listdir(folder)
        files = [f for f in files if f.endswith(".png")]
        l  = len(files)
        shutil.rmtree("results/iam_predictions", ignore_errors=True)
        if not os.path.exists("results/iam_predictions"):
            os.makedirs("results/iam_predictions")

        def process_file(file):
            image_path = os.path.join(folder, file)
            predicted_text = single_prediction(image_path, model, char_table, adapter)
            with open(f"results/iam_predictions/{file.split('.')[0]}.txt", "w") as f:
                f.write(f"{predicted_text}")

            return predicted_text

        with tqdm(total=l) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(process_file, arg): arg for arg in files}
                results = {}
                for future in concurrent.futures.as_completed(futures):
                    arg = futures[future]
                    results[arg] = future.result()
                    pbar.update(1)
            
    else:
        dataset_path = args.imagepath
        char_table_path = "char_table_single_im.txt"
        char_table = CharTable(char_table_path)

        model = HTRModel.create(model_path)

        adapter = model.get_adapter()
        pred = single_prediction(args.imagepath, model, char_table, adapter)

        print('Predicted: "{}"'.format(pred))

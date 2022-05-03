#%%
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras

import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
from utils import *

from pathlib import Path

np.random.seed(42)
tf.random.set_seed(42)

# %%
# Define defaults
# TODO: Change to args
main_path = Path("/media/hdd/github/handwriting_recog_codes/data/IAM-data/")
images_path = main_path / "img"
labels_path = main_path / "iam_lines_gt.txt"
batch_size = 200
image_size = (128, 32)
AUTOTUNE = tf.data.AUTOTUNE
#%%
iam_images, iam_labels = iam_data_reader(images_path, labels_path, image_size, subset=1000)
print(iam_images[:3], iam_labels[:3])

# %%

import concurrent
import os
import time
import urllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import *
from tqdm import tqdm
from PIL import Image
from tensorflow import keras
import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt
# from IPython.display import Image

import numpy as np
"""
This module contains all the general add ons
"""
def load_images_to_array(dss_path):
    """
    Return images and labels as numpy arrays
    """
    images,labels = [],[]
    for root, dirs, files in tqdm(os.walk(dss_path)):
        for file in files:
            if file.endswith(".pgm"):
                fname = os.path.join(root, file)
                images.append(np.array(Image.open(fname).convert("L").resize((28,28), Image.Resampling.BILINEAR))) #Biliear is good sampling
                # print(images.shape)
                labels.append(root.split("/")[-1]) #get the last folder name as label
    return images, labels

def label_to_dict(labels):
    labelmap= {label: i for i, label in enumerate(np.unique(labels))}
    return labelmap , [labelmap[label] for label in labels]

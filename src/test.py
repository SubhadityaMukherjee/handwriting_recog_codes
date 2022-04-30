# %%
import imutils
import cv2
import os
from PIL import Image, ImageFont, ImageDraw
from multiprocessing import Pool
from functools import partial
import numpy as np
from multiprocessing import Process, current_process
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import glob
from utils import *
import concurrent
import os
import time
import urllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

# %%
# Define defaults
# TODO: Change to args
main_path = Path(
    "/media/hdd/github/handwriting_recog_codes/data/image-data/image-data/")
# %%
list_of_binarized = [str(main_path/x)
                     for x in os.listdir(main_path) if "binarized" in x]
test_im = list_of_binarized[0]
# %%


image = cv2.imread(test_im)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2.imwrite('test_image.png',image)

#binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

#dilation
kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)

cv2.imwrite('dilated.png',img_dilation)

#find contours
ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = image[y:y+h, x:x+w]

    cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
    # TODO Add the model output here to these crops and get the output of it


cv2.imwrite('final_bounded_box_image.png',image)
# cv2.imshow('marked areas',image)
# cv2.waitKey(0)
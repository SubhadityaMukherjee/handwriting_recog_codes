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
import sys
# %%
# Define defaults
# TODO: Change to args
main_path = Path(
    "/Users/leozotos/Documents/Study/Master/1st Year/2B/HWR/handwriting_recog_codes/src/data/image-data")
# %%
list_of_binarized = [str(main_path/x)
                     for x in os.listdir(main_path) if "binarized" in x]
test_im = list_of_binarized[0]
# %%
raw_image = cv2.imread(test_im)
(thresh, image) = cv2.threshold(cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)

cv2.imwrite('test_image.png', image)
# %%
# Cutting to vertical stripes
numOfStripes = 2
numOfBlocks = 20 # that's the maximum number of lines
rowsPerBlock = 400 # max rows of pixels per block
pixelThreshold = 50
height, width = np.shape(image)
stripeWidth = width // numOfStripes
stripesArr = np.empty((numOfStripes, height, stripeWidth))

for i in range(numOfStripes): # Cut into stripes
    stripesArr[i] = image[:, (i*stripeWidth):(i+1)*stripeWidth]

allBlocks = np.full((numOfStripes, numOfBlocks, rowsPerBlock, stripeWidth), 255) # 4D array that will hold all blocks, set white background
for stripe in range(numOfStripes): 
    currentBlock = 0
    currentRowInBlock = 0
    for row_index, row in enumerate(stripesArr[stripe]):
        
        if np.count_nonzero(stripesArr[stripe, row_index] == 0) > pixelThreshold: # If this line has >10 black pixels, it is a line of text
            allBlocks[stripe, currentBlock, currentRowInBlock] = stripesArr[stripe, row_index]
            # cv2.imwrite("randomline.png", stripesArr[stripe, row])
            currentRowInBlock += 1
        else: # Create a new block, but only if the current block is an empty block.
            if not np.all(allBlocks[stripe, currentBlock] == 255):
                currentBlock += 1
                currentRowInBlock = 0


#Concatenate blocks: 
for line in range (numOfBlocks):
    concatenatedBlock = np.concatenate((allBlocks[0, line], allBlocks[1, line]), axis=1)
    if np.count_nonzero(concatenatedBlock == 0) > (2 * pixelThreshold): #Only store it if it has a lot of black pixels.
        linePath = 'lines/'
        nameOfOutput = "line" + str(line)+".png"
        cv2.imwrite(os.path.join(linePath , nameOfOutput), concatenatedBlock)
        
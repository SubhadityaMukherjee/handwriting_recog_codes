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
main_path = Path("data/image-data")
# %%
list_of_binarized = [str(main_path/x)
                     for x in os.listdir(main_path) if "binarized" in x]
test_im = list_of_binarized[0]
# %%
raw_image = cv2.imread(test_im)
(thresh, image) = cv2.threshold(cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
cv2.imwrite('test_image.png', image)

#################### keep only main text/ delete most white space around the text
cropPixelThreshold = 10
rowTopBoundary = rowBotBoundary = colLeftBoundary = colRightBoundary = 0

height, width = np.shape(image)
for rowIndex in range(height): # Remove top white space
    if np.count_nonzero(image[rowIndex] == 0) < cropPixelThreshold:
       rowTopBoundary = rowIndex
    else: break
image = np.delete(image, slice(0, rowTopBoundary), 0)

height, width = np.shape(image)
for rowIndex in range(height):# Remove top white space
    if np.count_nonzero(image[height-rowIndex - 1] == 0) < cropPixelThreshold:
        rowBotBoundary = rowIndex
    else: break
image = np.delete(image, slice(height-rowBotBoundary, height), 0)

height, width = np.shape(image)
for colIndex in range(width):# Remove left white space
    if np.count_nonzero(image[:, colIndex] == 0) < cropPixelThreshold:
       colLeftBoundary = colIndex
    else: break
image = np.delete(image, slice(0, colLeftBoundary), 1)

height, width = np.shape(image)
for colIndex in range(width):# Remove right white space
    if np.count_nonzero(image[:, width-colIndex - 1] == 0) < cropPixelThreshold:
        colRightBoundary = colIndex
    else: break
image = np.delete(image, slice(width-colRightBoundary, width), 1)

# ####################
# %%
# Cutting to vertical stripes
numOfStripes = 3
pixelThreshold = 30
numOfBlocks = 40 # that's the maximum number of lines
rowsPerBlock = 400 # max rows of pixels per block
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
if not os.path.exists("lines/"):
      os.makedirs("lines/")

for textLine in range (numOfBlocks):
    concatenatedBlock = allBlocks[0, textLine]
    for stripe in range(numOfStripes - 1):
        concatenatedBlock = np.concatenate((concatenatedBlock, allBlocks[stripe + 1, textLine]), axis = 1)

    if np.count_nonzero(concatenatedBlock == 0) > (numOfStripes * pixelThreshold): # Only store it if it has a lot of black pixels.
        linePath = 'lines/'
        nameOfOutput = "line" + str(textLine) + ".png"
        cv2.imwrite(os.path.join(linePath , nameOfOutput), concatenatedBlock)
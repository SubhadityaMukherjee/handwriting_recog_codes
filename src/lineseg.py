import os
import sys
from multiprocessing import Pool
from pathlib import Path
import argparse

import cv2
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

from utils import *


def importImagePaths(main_path):  # Return the paths to the images
    list_of_binarized_paths = [
        str(main_path / x) for x in os.listdir(main_path) if "binarized" in x
    ]
    return list_of_binarized_paths


# convert picture to black and white and dilate it (making characters fuller)
def blackWhiteDilate(imagePath):
    rawImage = cv2.imread(imagePath)

    kernel = np.ones((1, 1), np.uint8)
    # Even though we use erosion, the effect is more similar to dilation as the image is not inverted
    dilatedImage = cv2.erode(rawImage, kernel, iterations=1)

    (_, processedImage) = cv2.threshold(
        cv2.cvtColor(dilatedImage, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY
    )

    return processedImage


# Detects if the picture is slanted.
def isSlanted(image, threshold=0.125):
    hpp = np.sum(image, axis=1)
    normedHpp = hpp / hpp.mean()
    _, peaksInfo = find_peaks(normedHpp, height=0)

    if np.std(peaksInfo["peak_heights"]) < threshold:  # high: normal, low: slanted
        return True
    else:
        return False


def rotate_bound_white(image, angle):
    # Modified imutils.rotate_bound to have a white background
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(
        image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )


# Given an image, it tries to remove slanting by rotating it.
def fixRotation(image):
    bestImage = image
    stdOfBest = 0
    # Try different angles to find the one with the lowest std in peaks
    for angle in range(-10, 10):
        tempImage = rotate_bound_white(image, angle)
        hpp = np.sum(tempImage, axis=1)
        normedHpp = hpp / hpp.mean()
        _, peaksInfo = find_peaks(normedHpp, height=0)
        stdOfTemp = np.std(peaksInfo["peak_heights"])
        if stdOfTemp > stdOfBest:
            bestImage = tempImage
            stdOfBest = stdOfTemp
    return bestImage


# Keep only main text/ delete most white space around the text
def cropImage(
    image, threshold=5, all=True, top=False, bottom=False, left=False, right=False
):

    # Additional boolean arguments are used for croping only specific sides (and requites "all=False").
    rowTopBoundary = rowBotBoundary = colLeftBoundary = colRightBoundary = 0

    if all or top:
        height, width = np.shape(image)
        for rowIndex in range(height):  # Remove top white space
            if np.count_nonzero(image[rowIndex] == 0) < threshold:
                rowTopBoundary = rowIndex
            else:
                break
        image = np.delete(image, slice(0, rowTopBoundary), 0)

    if all or bottom:
        height, width = np.shape(image)
        for rowIndex in range(height):  # Remove bottom white space
            if np.count_nonzero(image[height - rowIndex - 1] == 0) < threshold:
                rowBotBoundary = rowIndex
            else:
                break
        image = np.delete(image, slice(height - rowBotBoundary, height), 0)

    if all or left:
        height, width = np.shape(image)
        for colIndex in range(width):  # Remove left white space
            if np.count_nonzero(image[:, colIndex] == 0) < threshold:
                colLeftBoundary = colIndex
            else:
                break
        image = np.delete(image, slice(0, colLeftBoundary), 1)

    if all or right:
        height, width = np.shape(image)
        for colIndex in range(width):  # Remove right white space
            if np.count_nonzero(image[:, width - colIndex - 1] == 0) < threshold:
                colRightBoundary = colIndex
            else:
                break
        image = np.delete(image, slice(width - colRightBoundary, width), 1)

    return image


# Make a folder for the lines of an image, and return the specific folder path of the current image
def makeImageFolder(folderName):
    specificFolderPath = os.path.basename(folderName)
    if not os.path.exists("lines/" + specificFolderPath):
        os.makedirs("lines/" + specificFolderPath)
    return specificFolderPath


# Returns an array with the stripes and the blocks in each stripe.
# tshPLow and tshPHigh determine the threshold for low and high density images respectively
# A higher threshold leads to more lines.
def createStripesAndBlocks(
    image,
    numOfStripes=1,
    tshPLow=3,
    tshPHigh=4.5,
    maxNumOfBlocks=40,
    maxRowsPerBlock=2000,
):
    """
    numOfStripes : The number of vertical stripes the image is cut into
    pxlThreshold : The number of black pixels that a row of pixels within a stripe needs to contain to be considered a text line
    maxNumOfBlocks : The maximum number of blocks of text per stripe (e.g. 30 blocks if you expect up to 30 lines of text, before under/oversegmentation is dealt with)
    maxRowsPerBlock : The maximum number of pixels that each text line might require.
    """
    height, width = np.shape(image)
    # Determine if picture is high or low density
    density = np.count_nonzero(image == 0) / np.count_nonzero(image == 255)
    threshold = 0
    if (
        density >= 0.12
    ):  # 0.12 experimentally found for dilation (1, 1). For dilation (2, 2) use 0.15
        # depending on resolution and density, use different threshold
        threshold = width / (tshPHigh * 10)
    else:
        # depending on resolution and density, use different threshold
        threshold = width / (tshPLow * 10)

    stripeWidth = width // numOfStripes
    stripesArr = np.empty((numOfStripes, height, stripeWidth))
    # used later to see if the padding should be at the top or bottom of the stripe (so that alignment with other stripes is correct). 0 for bottom padding, 1 for top padding
    padLocations = []
    for i in range(numOfStripes):  # Cut into stripes
        stripesArr[i] = image[:, (i * stripeWidth) : (i + 1) * stripeWidth]
        sliceSize = len(stripesArr[i]) // 20  # Take the bottom/top 5%
        if np.count_nonzero(
            stripesArr[i][0:sliceSize, 0 : len(stripesArr[i][0] - 1)] == 0
        ) > np.count_nonzero(stripesArr[i][len(stripesArr[i]) - sliceSize :, :] == 0):
            padLocations.append(0)  # Bot padding
        else:
            padLocations.append(1)  # Top padding
    # 4D array that will hold all blocks, set white background
    allBlocks = np.full(
        (numOfStripes, maxNumOfBlocks, maxRowsPerBlock, stripeWidth), 255
    )

    # Cut Stripe into blocks:
    for stripe in range(numOfStripes):
        currentBlock = 0
        currentRowInBlock = 0
        for row_index, row in enumerate(stripesArr[stripe]):
            # If this line has >10 black pixels, it is a line of text
            if np.count_nonzero(stripesArr[stripe, row_index] == 0) > threshold:
                allBlocks[stripe, currentBlock, currentRowInBlock] = stripesArr[
                    stripe, row_index
                ]
                currentRowInBlock += 1
            else:  # Create a new block, but only if the current block is an empty block.
                if not np.all(allBlocks[stripe, currentBlock] == 255):
                    currentBlock += 1
                    currentRowInBlock = 0
    return allBlocks, padLocations


# Deals with over/under segmentation of vertical blocks. Also removes blank space at the bottom of each block.
def overUnderSegmentation(stripesAndBlocks):
    adjustedStripesAndBlocks = []

    for stripeIndex, stripe in enumerate(stripesAndBlocks):
        blocksOfStripe = []
        averageBlockHeight = 0
        for block in stripe:
            cleanBlock = cropImage(block, 5, False, False, True)
            # More than 20 black pixels to be considered a text line block.
            if np.count_nonzero(cleanBlock == 0) > 50:
                blocksOfStripe.append(cleanBlock)

        averageBlockHeight = 0
        for block in blocksOfStripe:
            averageBlockHeight += len(block)
        averageBlockHeight /= len(blocksOfStripe)

        # Deal with Over-segmentation:
        blockIndex = 0
        while blockIndex < (len(blocksOfStripe)):
            if len(blocksOfStripe[blockIndex]) <= 0.5 * averageBlockHeight:
                # if it's the last block, combine it with the previous one.
                if blockIndex + 1 >= len(blocksOfStripe):
                    blocksOfStripe[blockIndex - 1] = np.concatenate(
                        (blocksOfStripe[blockIndex - 1], blocksOfStripe[blockIndex]),
                        axis=0,
                    )
                else:  # otherwise, combine it with the next block.
                    blocksOfStripe[blockIndex + 1] = np.concatenate(
                        (blocksOfStripe[blockIndex], blocksOfStripe[blockIndex + 1]),
                        axis=0,
                    )
                del blocksOfStripe[blockIndex]
                blockIndex -= 1
            blockIndex += 1

        # Re-calculate average line height after deadline with Over-segmentation
        averageBlockHeight = 0
        for block in blocksOfStripe:
            averageBlockHeight += len(block)
        averageBlockHeight /= len(blocksOfStripe)

        # Deal with Under-segmentation:
        # Assumption: Up to two lines are segmented together.
        blockIndex = 0
        while blockIndex < (len(blocksOfStripe)):
            if len(blocksOfStripe[blockIndex]) >= 1.5 * averageBlockHeight:
                middle = len(blocksOfStripe[blockIndex]) // 2
                firstHalf = blocksOfStripe[blockIndex][:middle]
                secondHalf = blocksOfStripe[blockIndex][middle:]
                del blocksOfStripe[blockIndex]
                blocksOfStripe.insert(blockIndex, firstHalf)
                blocksOfStripe.insert(blockIndex + 1, secondHalf)
            blockIndex += 1

        # Now create a new array with the adjusted blocks:
        adjustedStripesAndBlocks.insert(stripeIndex, blocksOfStripe)
    return adjustedStripesAndBlocks


# Make sure all blocks and stripes are of the same size by filling with blank space.
def standardiseBlocks(stripesAndBlocks, padLocations):
    maxNumberOfBlocks = 0
    maxNumberOfRowsPerBlock = 0
    blockWidth = len(stripesAndBlocks[0][0][0])
    for stripe in stripesAndBlocks:
        if len(stripe) > maxNumberOfBlocks:
            maxNumberOfBlocks = len(stripe)
        for block in stripe:
            if len(block) > maxNumberOfRowsPerBlock:
                maxNumberOfRowsPerBlock = len(block)

    standardisedBlocks = np.full(
        (len(stripesAndBlocks), maxNumberOfBlocks, maxNumberOfRowsPerBlock, blockWidth),
        255,
    )
    for stripeIndex in range(len(stripesAndBlocks)):
        # difference between height of the stripes
        heightDifference = len(standardisedBlocks[0]) - len(
            stripesAndBlocks[stripeIndex]
        )
        if padLocations[stripeIndex] == 0:  # pad bottom
            for blockIndex, block in enumerate(stripesAndBlocks[stripeIndex]):
                blockHeight = stripesAndBlocks[stripeIndex][blockIndex].shape[0]
                standardisedBlocks[stripeIndex][blockIndex][
                    0:blockHeight, 0:blockWidth
                ] = block
        else:  # pad top
            for blockIndex, block in enumerate(stripesAndBlocks[stripeIndex]):
                blockHeight = stripesAndBlocks[stripeIndex][blockIndex].shape[0]
                standardisedBlocks[stripeIndex][blockIndex + heightDifference][
                    0:blockHeight, 0:blockWidth
                ] = block

    return standardisedBlocks


# Concatenates blocks and exports the lines to their folder.
def concatBlocksAndExport(stripesAndBlocks, exportPath):
    numOfStripes = len(stripesAndBlocks)
    numOfBlocks = stripesAndBlocks.shape[1]
    for textLine in range(numOfBlocks):
        concatenatedBlock = stripesAndBlocks[0, textLine]
        for stripe in range(numOfStripes - 1):
            concatenatedBlock = np.concatenate(
                (concatenatedBlock, stripesAndBlocks[stripe + 1, textLine]), axis=1
            )

        linePath = "lines/" + exportPath
        nameOfOutput = str(textLine) + ".png"
        cv2.imwrite(os.path.join(linePath, nameOfOutput), concatenatedBlock)

    return stripesAndBlocks


def stripeSegmentation(image, folderPath):
    # Splits the image into stripes and blocks per stripe
    stripesAndBlocks, padLocations = createStripesAndBlocks(image)
    # Merges or splits blocks, dealing with over/under block segmentation
    stripesAndBlocks = overUnderSegmentation(stripesAndBlocks)
    # Makes sure stripes and blocks are the same size, by filling with blank space
    stripesAndBlocks = standardiseBlocks(stripesAndBlocks, padLocations)
    concatBlocksAndExport(stripesAndBlocks, folderPath)


def contourSegmentation(image, folderPath):
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    # dilation
    kernel = np.ones((5, 100), np.uint8)
    # 5,100 for line, 5.5 for character
    img_dilation = cv2.dilate(image, kernel, iterations=1)

    # find contours
    ctrs, _ = cv2.findContours(
        img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = image[y : y + h, x : x + w]
        linePath = "lines/" + folderPath
        nameOfOutput = str(i) + ".png"
        if len(roi[0]) > 500:  # Don't save very small contours
            cv2.imwrite(os.path.join(linePath, nameOfOutput), roi)


if __name__ == "__main__":
    ag = argparse.ArgumentParser()
    ag.add_argument("-f", type=str, help="Path to images", default="../data/image-data/image-data")
    ag.add_argument("--type", type=int, help="Type of segmentation", default=0)
    ap = ag.parse_args()

    imagePaths = importImagePaths(Path(ap.f))
    # Make general lines folder for the output of each image
    if not os.path.exists("lines/"):
        os.makedirs("lines/")

    for imageIndex, imagePath in tqdm(enumerate(imagePaths), total = len(imagePaths)):
        # print(
        #     "Line Segmentation Progress: "
        #     + str(int(imageIndex * 100 / len(imagePaths)))
        #     + "%"
        # )
        # Folders will be created inside the general Lines folder
        folderPath = makeImageFolder(str(imagePath))
        # Make the picture black and white and crop it so that only the text is present.
        image = fixRotation(cropImage(blackWhiteDilate(imagePath), 15))
        if ap.type == 0:
            stripeSegmentation(image, folderPath)
        elif ap.type == 1:
            contourSegmentation(image, folderPath)
        else:
            sys.exit("Argument " + "'" + str(sys.argv[1]) + "' not recognised")

    print("Line Segmentation Complete!")

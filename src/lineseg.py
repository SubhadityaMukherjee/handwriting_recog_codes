import cv2
import os
from multiprocessing import Pool
import numpy as np
from pathlib import Path
from utils import *
import os
from pathlib import Path


def importImagePaths(main_path):  # Return the paths to the images
    list_of_binarized_paths = [str(main_path/x)
                               for x in os.listdir(main_path) if "binarized" in x]
    return list_of_binarized_paths


def blackAndWhite(imagePath):  # convert picture to black and white
    rawImage = cv2.imread(imagePath)
    (_, processedImage) = cv2.threshold(cv2.cvtColor(
        rawImage, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)

    return processedImage


# Keep only main text/ delete most white space around the text
def cropImage(image, threshold=10, all=True, top=False, bottom=False, left=False, right=False):
    # Additional boolean arguments are used for croping only specific sides (and requites "all=False").
    rowTopBoundary = rowBotBoundary = colLeftBoundary = colRightBoundary = 0

    if(all or top):
        height, width = np.shape(image)
        for rowIndex in range(height):  # Remove top white space
            if np.count_nonzero(image[rowIndex] == 0) < threshold:
                rowTopBoundary = rowIndex
            else:
                break
        image = np.delete(image, slice(0, rowTopBoundary), 0)

    if(all or bottom):
        height, width = np.shape(image)
        for rowIndex in range(height):  # Remove bottom white space
            if np.count_nonzero(image[height-rowIndex - 1] == 0) < threshold:
                rowBotBoundary = rowIndex
            else:
                break
        image = np.delete(image, slice(height-rowBotBoundary, height), 0)

    if(all or left):
        height, width = np.shape(image)
        for colIndex in range(width):  # Remove left white space
            if np.count_nonzero(image[:, colIndex] == 0) < threshold:
                colLeftBoundary = colIndex
            else:
                break
        image = np.delete(image, slice(0, colLeftBoundary), 1)

    if(all or right):
        height, width = np.shape(image)
        for colIndex in range(width):  # Remove right white space
            if np.count_nonzero(image[:, width-colIndex - 1] == 0) < threshold:
                colRightBoundary = colIndex
            else:
                break
        image = np.delete(image, slice(width-colRightBoundary, width), 1)

    return image


# Make a folder for the lines of an image, and return the specific folder path of the current image
def makeImageFolder(folderName):
    specificFolderPath = os.path.basename(folderName)
    if not os.path.exists("lines/" + specificFolderPath):
        os.makedirs("lines/" + specificFolderPath)
    return specificFolderPath


# Returns an array with the stripes and the blocks in each stripe
def createStripesAndBlocks(image, numOfStripes=2, pxlThreshold=15, maxNumOfBlocks=40, maxRowsPerBlock=1000):
    """
    numOfStripes : The number of vertical stripes the image is cut into
    pxlThreshold : The number of black pixels that a row of pixels within a stripe needs to contain to be considered a text line
    maxNumOfBlocks  : The maximum number of blocks of text per stripe (e.g. 30 blocks if you expect up to 30 lines of text, before under/oversegmentation is dealt with)
    maxRowsPerBlock  : The maximum number of pixels that each text line might require.
    """

    height, width = np.shape(image)
    stripeWidth = width // numOfStripes
    stripesArr = np.empty((numOfStripes, height, stripeWidth))

    for i in range(numOfStripes):  # Cut into stripes
        stripesArr[i] = image[:, (i*stripeWidth):(i+1)*stripeWidth]
    # 4D array that will hold all blocks, set white background
    allBlocks = np.full((numOfStripes, maxNumOfBlocks,
                        maxRowsPerBlock, stripeWidth), 255)

    # Cut Stripe into blocks:
    for stripe in range(numOfStripes):
        currentBlock = 0
        currentRowInBlock = 0
        for row_index, row in enumerate(stripesArr[stripe]):
            # If this line has >10 black pixels, it is a line of text
            if np.count_nonzero(stripesArr[stripe, row_index] == 0) > pxlThreshold:
                allBlocks[stripe, currentBlock,
                          currentRowInBlock] = stripesArr[stripe, row_index]
                currentRowInBlock += 1
            else:  # Create a new block, but only if the current block is an empty block.
                if not np.all(allBlocks[stripe, currentBlock] == 255):
                    currentBlock += 1
                    currentRowInBlock = 0
    return allBlocks


# Deals with over/under segmentation of vertical blocks. Also removes blank space at the bottom of each block.
def overUnderSegmentation(stripesAndBlocks):
    adjustedStripesAndBlocks = []

    for stripeIndex, stripe in enumerate(stripesAndBlocks):
        blocksOfStripe = []
        averageBlockHeight = 0
        for block in stripe:
            cleanBlock = cropImage(block, 2, False, False, True)
            # More than 5 black pixels to be considered a text line block.
            if np.count_nonzero(cleanBlock == 0) > 20:
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
                if(blockIndex + 1 >= len(blocksOfStripe)):
                    blocksOfStripe[blockIndex - 1] = np.concatenate(
                        (blocksOfStripe[blockIndex-1], blocksOfStripe[blockIndex]), axis=0)
                else:  # otherwise, combine it with the next block.
                    blocksOfStripe[blockIndex + 1] = np.concatenate(
                        (blocksOfStripe[blockIndex], blocksOfStripe[blockIndex+1]), axis=0)
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
            if len(blocksOfStripe[blockIndex]) > 1.5 * averageBlockHeight:
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
def standardiseBlocks(stripesAndBlocks):
    maxNumberOfBlocks = 0
    maxNumberOfRowsPerBlock = 0
    blockWidth = len(stripesAndBlocks[0][0][0])
    for stripe in stripesAndBlocks:
        if len(stripe) > maxNumberOfBlocks:
            maxNumberOfBlocks = len(stripe)
        for block in stripe:
            if len(block) > maxNumberOfRowsPerBlock:
                maxNumberOfRowsPerBlock = len(block)
    
    standardisedBlocks = np.full((len(stripesAndBlocks), maxNumberOfBlocks, maxNumberOfRowsPerBlock, blockWidth), 255)
    for stripeIndex in range(len(stripesAndBlocks)):
        for blockIndex, block in enumerate(stripesAndBlocks[stripeIndex]):
            blockHeight = stripesAndBlocks[stripeIndex][blockIndex].shape[0]
            standardisedBlocks[stripeIndex][blockIndex][0:blockHeight, 0:blockWidth] = block

    return standardisedBlocks


# Concatenates blocks and exports the lines to their folder.
def concatBlocksAndExport(stripesAndBlocks, exportPath):
    numOfStripes = len(stripesAndBlocks)
    numOfBlocks = stripesAndBlocks.shape[1]
    for textLine in range(numOfBlocks):
        concatenatedBlock = stripesAndBlocks[0, textLine]
        for stripe in range(numOfStripes - 1):
            concatenatedBlock = np.concatenate(
                (concatenatedBlock, stripesAndBlocks[stripe + 1, textLine]), axis=1)

        linePath = 'lines/' + exportPath
        nameOfOutput = str(textLine) + ".png"
        cv2.imwrite(os.path.join(linePath, nameOfOutput), concatenatedBlock)

    return stripesAndBlocks


if __name__ == "__main__":
    imagePaths = importImagePaths(Path("data/image-data"))
    # Make general lines folder for the output of each image
    if not os.path.exists("lines/"):
        os.makedirs("lines/")

    # Segment Lines for each Image
    for imageIndex, imagePath in enumerate(imagePaths):
        print("Line Segmentation Progress: " +
              str(imageIndex*100/len(imagePaths)) + "%")
        # Folders will be created inside the general Lines folder
        folderPath = makeImageFolder(str(imagePath))
        # Make the picture black and white and crop it so that only the text is present.
        image = cropImage(blackAndWhite(imagePath), 20)
        # Splits the image into stripes and blocks per stripe
        stripesAndBlocks = createStripesAndBlocks(image)
        # Merges or splits blocks, dealing with over/under block segmentation
        stripesAndBlocks = overUnderSegmentation(stripesAndBlocks)
        #Makes sure stripes and blocks are the same size, by filling with blank space
        stripesAndBlocks = standardiseBlocks(stripesAndBlocks)
        concatBlocksAndExport(stripesAndBlocks, folderPath)

    print("Line Segmentation Complete!")

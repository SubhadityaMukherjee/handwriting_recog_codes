import cv2
import os
from multiprocessing import Pool
import numpy as np
from pathlib import Path
from utils import *
import os
from pathlib import Path

def importImagePaths(main_path): # Return the paths to the images
    list_of_binarized_paths = [str(main_path/x)
                        for x in os.listdir(main_path) if "binarized" in x]
    return list_of_binarized_paths

def blackAndWhite(imagePath): # convert picture to black and white
    rawImage = cv2.imread(imagePath)
    (_, processedImage) = cv2.threshold(cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    
    return processedImage

def cropImage(image, threshold = 10): # Keep only main text/ delete most white space around the text
    rowTopBoundary = rowBotBoundary = colLeftBoundary = colRightBoundary = 0

    height, width = np.shape(image)
    for rowIndex in range(height): # Remove top white space
        if np.count_nonzero(image[rowIndex] == 0) < threshold:
            rowTopBoundary = rowIndex
        else: break
    image = np.delete(image, slice(0, rowTopBoundary), 0)

    height, width = np.shape(image)
    for rowIndex in range(height):# Remove top white space
        if np.count_nonzero(image[height-rowIndex - 1] == 0) < threshold:
            rowBotBoundary = rowIndex
        else: break
    image = np.delete(image, slice(height-rowBotBoundary, height), 0)

    height, width = np.shape(image)
    for colIndex in range(width):# Remove left white space
        if np.count_nonzero(image[:, colIndex] == 0) < threshold:
            colLeftBoundary = colIndex
        else: break
    image = np.delete(image, slice(0, colLeftBoundary), 1)

    height, width = np.shape(image)
    for colIndex in range(width):# Remove right white space
        if np.count_nonzero(image[:, width-colIndex - 1] == 0) < threshold:
            colRightBoundary = colIndex
        else: break
    image = np.delete(image, slice(width-colRightBoundary, width), 1)
    return image

def makeImageFolder(folderName): # Make a folder for the lines of an image, and return the specific folder path of the current image
    specificFolderPath = os.path.basename(folderName)
    if not os.path.exists("lines/" + specificFolderPath):
        os.makedirs("lines/" + specificFolderPath)
    return specificFolderPath

def createStripesAndBlocks(image, numOfStripes = 3, pxlThreshold = 10, maxNumOfBlocks = 40, maxRowsPerBlock = 1000): # Returns an array with the stripes and the blocks in each stripe
    """
    numOfStripes : The number of vertical stripes the image is cut into
    pxlThreshold : The number of black pixels that a row of pixels within a stripe needs to contain to be considered a text line
    maxNumOfBlocks  : The maximum number of blocks of text per stripe (e.g. 30 blocks if you expect up to 30 lines of text, before under/oversegmentation is dealt with)
    maxRowsPerBlock  : The maximum number of pixels that each text line might require.
    """

    height, width = np.shape(image)
    stripeWidth = width // numOfStripes
    stripesArr = np.empty((numOfStripes, height, stripeWidth))

    for i in range(numOfStripes): # Cut into stripes
        stripesArr[i] = image[:, (i*stripeWidth):(i+1)*stripeWidth]
    allBlocks = np.full((numOfStripes, maxNumOfBlocks, maxRowsPerBlock, stripeWidth), 255) # 4D array that will hold all blocks, set white background

    # Cut Stripe into blocks:
    for stripe in range(numOfStripes): 
        currentBlock = 0
        currentRowInBlock = 0
        for row_index, row in enumerate(stripesArr[stripe]):
            
            if np.count_nonzero(stripesArr[stripe, row_index] == 0) > pxlThreshold: # If this line has >10 black pixels, it is a line of text
                allBlocks[stripe, currentBlock, currentRowInBlock] = stripesArr[stripe, row_index]
                currentRowInBlock += 1
            else: # Create a new block, but only if the current block is an empty block.
                if not np.all(allBlocks[stripe, currentBlock] == 255):
                    currentBlock += 1
                    currentRowInBlock = 0
    return allBlocks

def concatBlocksAndExport(stripesAndBlocks, exportPath, lineThreshold = 20): # Concatenates blocks and exports the lines to their folder.
    # lineThreshold : the number of black pixels a concatenated line needs to have to be considered a line (instead of a white line for example)
    numOfStripes = stripesAndBlocks.shape[0]
    numOfBlocks = stripesAndBlocks.shape[1]
    for textLine in range (numOfBlocks):
        concatenatedBlock = stripesAndBlocks[0, textLine]
        for stripe in range(numOfStripes - 1):
            concatenatedBlock = np.concatenate((concatenatedBlock, stripesAndBlocks[stripe + 1, textLine]), axis = 1)

        if np.count_nonzero(concatenatedBlock == 0) > (lineThreshold): # Only store it if it has a lot of black pixels.
            linePath = 'lines/' + exportPath
            nameOfOutput = str(textLine) + ".png"
            cv2.imwrite(os.path.join(linePath , nameOfOutput), concatenatedBlock)

def overUnderSegmentation(stripesAndBlocks):
    print("under/over segmentation still needs to be dealt with!")
    return stripesAndBlocks

if __name__ == "__main__":
    imagePaths = importImagePaths(Path("data/image-data"))
    if not os.path.exists("lines/"): # Make general lines folder for the output of each image
        os.makedirs("lines/")

    for imageIndex, imagePath in enumerate(imagePaths): # Segment Lines for each Image
        print("Line Segmentation Progress: "+ str(imageIndex*100/len(imagePaths))+ "%")
        folderPath = makeImageFolder(str(imagePath)) # Folders will be created inside the general Lines folder
        image = cropImage(blackAndWhite(imagePath), 20) # Make the picture black and white and crop it so that only the text is present. 
        stripesAndBlocks = createStripesAndBlocks(image) # Splits the image into stripes and blocks per stripe
        stripesAndBlocks = overUnderSegmentation(stripesAndBlocks) # Merges or splits blocks, dealing with over/under block segmentation
        concatBlocksAndExport(stripesAndBlocks, folderPath)

    print("Line Segmentation Complete!")
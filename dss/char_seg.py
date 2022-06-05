import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils import *

"""
The following function splits connected elements
"""


def splitConnectedElements(image, box):
    newBoxes = []

    # Focus is set on the connected elements
    focus = (
            255 - image[(box[1]):(box[1] + box[3] + 2), (box[0]):(box[0] + box[2] + 2)]
    )
    # Black and dark grey pixels are set to 1 and white and very light gray pixels are set to 0
    focus[focus <= 127] = 1
    focus[focus > 127] = 0

    # Vertical projection is performed
    projection = np.sum(focus, axis=0)

    # Width to height ration is calculated. And rounded up or down based on the below thresholds
    # The ratio approximates how many characters should be in the box.
    if (((box[2] / box[3]) % 1) >= .35) and ((box[2] / box[3]) < 2):
        ratio = int(box[2] / box[3]) + 1
    elif ((box[2] / box[3]) % 1) >= .23:
        ratio = int(box[2] / box[3]) + 1
    else:
        ratio = int(box[2] / box[3])

    # Following if statement splits two connected characters
    if ratio == 2:
        # The minimum projection value from the middle 50% of the image is taken
        localMinimum = min(projection[(int(box[2] * 0.25)):(int(box[2] * 0.75))])

        # The split point is taken based on the above minimum projection
        splitPoint = (int(box[2] * 0.25)) + \
                     np.where(projection[(int(box[2] * 0.25)):(int(box[2] * 0.75))] == localMinimum)[0][0]

        # The split only happens if both boxes' widths are bigger than 20 pixels.
        # If one box would be under 20, that box would bound noise or an element of the character
        if splitPoint >= 20:
            newBoxes.append((box[0], box[1], splitPoint + 3, box[3]))
        if (box[2] - splitPoint) >= 20:
            newBoxes.append((box[0] + splitPoint - 3, box[1], box[2] - splitPoint, box[3]))

    # Following if statements split three or more connected elements
    elif ratio == 3:

        # A preliminary spli point is taken as the right most margin
        splitPoint = box[2]

        # The "step" shows approximately what the width of a character should be (based on the ratio)
        step = int(box[2] / ratio)

        # Iterating through how many splits should happen
        for i in range(1, ratio):
            # The margins between which the split should happen are declared.
            # The margins are set around the point where a character theoretically should end in comparison to
            # where is starts (based on the ratio).
            # The margins delimit 30% of the connected character box's width in which a split should happen
            # (30% is only for three connected elements a higher number of connected elements
            # requires a smaller percentage).
            lowerMargin = splitPoint - step - int(box[2] * 0.15)
            if lowerMargin < 0:
                lowerMargin = 4
            upperMargin = splitPoint - step + int(box[2] * 0.15)

            # The minimum projection between the margins is taken
            localMinimum = min(projection[lowerMargin:upperMargin])

            # The previous split point (right margin of character) is saved and new split point is
            # calculated based on the minimum projection
            prevSplit = splitPoint
            splitPoint = lowerMargin + np.where(projection[lowerMargin:upperMargin] == localMinimum)[0][0]

            # If the box's width is bigger than 20 pixels (the box does not bound noise or
            # just a simple element of a character), the split happens.
            if prevSplit - splitPoint >= 20:
                if prevSplit == box[2]:
                    newBoxes.append((box[0] + splitPoint - 3, box[1], prevSplit - splitPoint, box[3]))
                else:
                    newBoxes.append((box[0] + splitPoint - 3, box[1], prevSplit - splitPoint + 3, box[3]))
        if splitPoint >= 20:
            newBoxes.append((box[0], box[1], splitPoint + 3, box[3]))

        # Order of the boxes is reversed to be in the correct order for appending in the general bounding boxes' list
        newBoxes.reverse()

    # Following if statements perform the exact same procedures but the lower and upper margin between which the
    # split should happen offer different margins.
    # For 4 connected elements 30% of the width of the image is in focus at one time for splitting.
    # For 5 connected elements 20% of the width of the image is in focus at one time for splitting.
    elif ratio == 4:
        splitPoint = box[2]
        step = int(box[2] / ratio)
        for i in range(1, ratio):
            lowerMargin = splitPoint - step - int(box[2] * 0.15)
            if lowerMargin < 0:
                lowerMargin = 4
            upperMargin = splitPoint - step + int(box[2] * 0.15)
            localMinimum = min(projection[lowerMargin:upperMargin])
            prevSplit = splitPoint
            splitPoint = lowerMargin + np.where(projection[lowerMargin:upperMargin] == localMinimum)[0][0]
            if prevSplit - splitPoint >= 20:
                if prevSplit == box[2]:
                    newBoxes.append((box[0] + splitPoint - 3, box[1], prevSplit - splitPoint, box[3]))
                else:
                    newBoxes.append((box[0] + splitPoint - 3, box[1], prevSplit - splitPoint + 3, box[3]))
        if splitPoint >= 20:
            newBoxes.append((box[0], box[1], splitPoint + 3, box[3]))
        newBoxes.reverse()
    elif ratio == 5:
        splitPoint = box[2]
        step = int(box[2] / ratio)
        for i in range(1, ratio):
            lowerMargin = splitPoint - step - int(box[2] * 0.10)
            if lowerMargin < 0:
                lowerMargin = 4
            upperMargin = splitPoint - step + int(box[2] * 0.10)
            localMinimum = min(projection[lowerMargin:upperMargin])

            prevSplit = splitPoint
            splitPoint = lowerMargin + np.where(projection[lowerMargin:upperMargin] == localMinimum)[0][0]
            if prevSplit - splitPoint >= 20:
                if prevSplit == box[2]:
                    newBoxes.append((box[0] + splitPoint - 3, box[1], prevSplit - splitPoint, box[3]))
                else:
                    newBoxes.append((box[0] + splitPoint - 3, box[1], prevSplit - splitPoint + 3, box[3]))
        if splitPoint >= 20:
            newBoxes.append((box[0], box[1], splitPoint + 3, box[3]))
        newBoxes.reverse()
    else:
        newBoxes.append(box)
    return newBoxes


"""
The following function merges two bounding boxes into one
"""


def mergeBoxes(box1, box2):
    holder = []

    # The top most and left most coordinates from either one of the two bounding boxes
    # are taken as the new coordinates
    p1 = min(box1[0], box2[0])
    p2 = min(box1[1], box2[1])

    # The bottom most and right most coordinates from either one of the two bounding boxes
    # are taken as the new coordinates
    p3 = max(box1[2], box2[2]) + abs(box1[0] - box2[0])
    p4 = max(box1[3], box2[3]) + abs(box1[1] - box2[1])
    holder.append(p1)
    holder.append(p2)
    holder.append(p3)
    holder.append(p4)
    holder = tuple(holder)

    # The new bounding box is returned
    return holder


"""
The following function checks whether two boxes need to be merged
"""

def checkMerge(box1, box2):
    # Following lines merge two bounding boxes, if box1 is inside box2.
    # A leeway is given, so box1 can be 70% of its width outside box2 (on the x axis)
    if ((box2[0] - box1[2] * 7 / 10) <= box1[0]) and (
            (box2[0] + box2[2] + box1[2] * 7 / 10) >= (box1[0] + box1[2])
    ):
        # Following if statement is for the y axis. On this axis, if at least one of the edges of the current
        # bounding box is inside the previous bounding box the boxes will be merged.
        if (box2[1] < (box1[1] + box1[3]) <= (box2[1] + box2[3])) or (
                (box2[1] + box2[3]) > box1[1] >= box2[1]
        ):
            # The two bounding boxes are merged and the new combined box is returned
            box1 = mergeBoxes(box1, box2)
            return box1
    # If the above conditions are not met, the boxes are left the way they were
    else:
        return


"""
The following function is used to clean and improve the bounding boxes.
The boxes are improved by merging, splitting or ignoring certain boxes, based on discovered thresholds.
"""

def cleanBoxes(image, box, bbox):
    # Bounding box is saved only if it is 25 pixels or bigger
    # This eliminates small bounding boxes on noise
    if (box[2] < 25) or (box[3] < 25):
        return

    # If the box's width is bigger then 40 pixels and its width to height ration is below 1.2,
    # it means that the box bounds just a single character.
    # These characters are directly returned
    elif (box[2] > 40) and (box[2] / box[3] < 1.2):
        return box

    # If the width to height ratio of the box is bigger than 1.2,
    # it means that the box bounds 2 or more connected characters.
    # These characters are split
    elif box[2] / box[3] >= 1.2:
        box = splitConnectedElements(image, box)
        return box

    # Other boxes (usually between 25 and 40 pixels) might bound elements of characters which
    # shouldn't be independent.
    # If this if the case the elements are merged with the character.
    # The procedure is performed to up to two boxes behind the current box.
    else:
        if len(bbox) >= 2:
            hold = checkMerge(bbox[-2], box)
            if hold:
                bbox.pop()
                bbox.append(hold)

        if len(bbox) >= 1:
            hold = checkMerge(box, bbox[-1])
            if hold:
                bbox.pop()
                bbox.append(hold)
            hold = checkMerge(bbox[-1], box)
            if hold:
                bbox.pop()
                bbox.append(hold)
        return box


"""
The following function is used to get all the bounding boxes for the characters in a line.
"""


def getBBox(image):
    # Getting characters' contours and creating a hierarchy
    cnt, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    poly = [None] * len(cnt)

    # Declaring bounding box list
    bbox = []

    # Bounding boxes are created
    for i, contour in enumerate(cnt):
        poly[i] = cv2.approxPolyDP(contour, 3, closed=True)

        if hierarchy[0][i][3] == -1:
            box = cv2.boundingRect(poly[i])

            # Bounding boxes are cleaned and improved
            box = cleanBoxes(image, box, bbox)

            # Each bounding box is appended to the list
            if box and (type(box) is tuple):
                bbox.append(box)
            elif type(box) is list:
                for item in box:
                    bbox.append(item)

    return bbox


"""
The following function takes in the path to one line in an image, 
uses the functions above to segment the characters, then it outputs the characters to their corresponding folders
"""


def oneLineProcessing(img_str, line):
    linePath = "lines/" + str(img_str) + "/" + str(line)

    # Folders for each line are created.
    # If the number of the line is below 10, a 0 is added to the name of the folder (for sorting purposes)
    if int(line[:-4]) <= 9:
        if not os.path.exists("lines/" + img_str + "/characters/line0" + str(line[:-4])):
            os.makedirs("lines/" + img_str + "/characters/line0" + str(line[:-4]))
    else:
        if not os.path.exists("lines/" + img_str + "/characters/line" + str(line[:-4])):
            os.makedirs("lines/" + img_str + "/characters/line" + str(line[:-4]))

    # Image is created from the line path and it is preprocessed using the "preprocess" function from utils.py
    img = cv2.imread(linePath)
    processed = preprocess(img)

    # Getting the bounding boxes from the inverted version of the processed image
    bbox = getBBox(255 - processed)
    # Sorting the bounding boxes based on their left corner x-coordinate (in an ascending order)
    bbox.sort()

    for i in range(len(bbox)):

        # The borders of the bounding box are declared for readability
        # and they are then used to select each character from the line
        corner1, corner2, corner3, corner4 = (
            int(bbox[i][0]),
            int(bbox[i][1]),
            int(bbox[i][2]),
            int(bbox[i][3]),
        )
        focus = img[corner2:(corner2 + corner4), corner1:(corner1 + corner3)].copy()

        # The character images are outputted to their corresponding folders
        # If the index of the character is below 10 a "0" will be added in front of the index,
        # in the image name(for sorting purposes)
        if focus.any():
            if int(line[:-4]) <= 9:
                if i <= 9:
                    cv2.imwrite("lines/" + str(img_str) + "/characters/line0" + str(line[:-4]) + "/0" + str(i) + ".png",
                                focus)
                else:
                    cv2.imwrite("lines/" + str(img_str) + "/characters/line0" + str(line[:-4]) + "/" + str(i) + ".png",
                                focus)
            else:
                if i <= 9:
                    cv2.imwrite("lines/" + str(img_str) + "/characters/line" + str(line[:-4]) + "/0" + str(i) + ".png",
                                focus)
                else:
                    cv2.imwrite("lines/" + str(img_str) + "/characters/line" + str(line[:-4]) + "/" + str(i) + ".png",
                                focus)


"""
The following function goes through each of the lines in the images' folders(under the "lines" folder) and processes them
"""


def charSegmentation(imagesPath):
    for img in tqdm(imagesPath):
        if img != ".DS_Store":
            imgPath = os.listdir("lines/" + str(img) + "/")
            for line in imgPath:
                if str(line) == "characters":
                    break
                oneLineProcessing(img_str=img, line=line)


if __name__ == "__main__":
    print("\nSegmenting Characters:")
    imagesPath = os.listdir("lines/")
    charSegmentation(imagesPath)

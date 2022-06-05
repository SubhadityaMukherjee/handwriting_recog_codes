import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils import *


def mergeBoxes(box1, box2):
    holder = []
    p1 = min(box1[0], box2[0])
    p2 = min(box1[1], box2[1])
    p3 = max(box1[2], box2[2]) + abs(box1[0] - box2[0])
    p4 = max(box1[3], box2[3]) + abs(box1[1] - box2[1])
    holder.append(p1)
    holder.append(p2)
    holder.append(p3)
    holder.append(p4)
    holder = tuple(holder)
    return holder


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
            box1 = mergeBoxes(box1, box2)
            return box1
    else:
        return


def splitConnectedElements(image, box):
    newBoxes = []
    focus = (
            255 - image[(box[1]):(box[1] + box[3] + 2), (box[0]):(box[0] + box[2] + 2)]
    )
    copy = focus.copy()
    focus[focus <= 127] = 1
    focus[focus > 127] = 0
    projection = np.sum(focus, axis=0)
    if (((box[2] / box[3]) % 1) >= .35) and ((box[2] / box[3]) < 2):
        ratio = int(box[2] / box[3]) + 1
    elif ((box[2] / box[3]) % 1) >= .23:
        ratio = int(box[2] / box[3]) + 1
    else:
        ratio = int(box[2] / box[3])
    if ratio == 2:
        localMinimum = min(projection[(int(box[2] * 0.25)):(int(box[2] * 0.75))])
        splitPoint = (int(box[2] * 0.25)) + \
                     np.where(projection[(int(box[2] * 0.25)):(int(box[2] * 0.75))] == localMinimum)[0][0]
        if splitPoint >= 20:
            newBoxes.append((box[0], box[1], splitPoint + 3, box[3]))
        if (box[2] - splitPoint) >= 20:
            newBoxes.append((box[0] + splitPoint - 3, box[1], box[2] - splitPoint, box[3]))
    elif ratio == 3:
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


def cleanBoxes(image, box, bbox):
    # Bounding box is saved only if it is 25 pixels or bigger
    # This eliminates small bounding boxes on noise
    if (box[2] < 25) or (box[3] < 25):
        return
    # 1 character is somewhere between 25-40 pixels and 75 pixels in width
    # These characters are directly returned
    elif (box[2] > 40) and (box[2] / box[3] < 1.2):
        return box
    # Taking care of connected components
    elif box[2] / box[3] >= 1.2:
        box = splitConnectedElements(image, box)
        return box
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
        # Following lines merge two bounding boxes, if the current bounding box is inside the previous registered
        # bounding box. A leeway is given, so the current bounding bounding box can be 70% of its width outside the
        # previous bounding box (on the x axis)
        return box


def getBBox(image):
    # Getting characters' contours and creating a hierarchy
    cnt, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    poly = [None] * len(cnt)

    # Declaring bounding box list
    bbox = []

    for i, contour in enumerate(cnt):
        poly[i] = cv2.approxPolyDP(contour, 3, closed=True)

        if hierarchy[0][i][3] == -1:
            box = cv2.boundingRect(poly[i])
            box = cleanBoxes(image, box, bbox)

            if box and (type(box) is tuple):
                bbox.append(box)
            elif type(box) is list:
                for item in box:
                    bbox.append(item)

    return bbox


def oneLineProcessing(img_str, line):
    linePath = "lines/" + str(img_str) + "/" + str(line)
    if int(line[:-4]) <= 9:
        if not os.path.exists("lines/" + img_str + "/characters/line0" + str(line[:-4])):
            os.makedirs("lines/" + img_str + "/characters/line0" + str(line[:-4]))
    else:
        if not os.path.exists("lines/" + img_str + "/characters/line" + str(line[:-4])):
            os.makedirs("lines/" + img_str + "/characters/line" + str(line[:-4]))
    img = cv2.imread(linePath)
    processed = preprocess(img)

    # Getting the bounding boxes from the inverted version of the processed image
    bbox = getBBox(255 - processed)
    # Sorting the bounding boxes based on their left corner x-coordinate
    bbox.sort()

    for i in range(len(bbox)):
        corner1, corner2, corner3, corner4 = (
            int(bbox[i][0]),
            int(bbox[i][1]),
            int(bbox[i][2]),
            int(bbox[i][3]),
        )
        focus = img[corner2:(corner2 + corner4), corner1:(corner1 + corner3)].copy()
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

"""
def charSegmentation(imagesPath):
    for img in tqdm(imagesPath):
        if img != ".DS_Store":
            imgPath = os.listdir("lines/" + str(img) + "/")
            for line in imgPath:
                if (str(img) == "x") or (str(img) == "connected"):
                    break
                elif str(img) == "bbox" or str(line) == "characters":
                    break
                oneLineProcessing(img_str=img, line=line)


if __name__ == "__main__":
    imagesPath = os.listdir("lines/")
    print("\nSegmenting Characters:")
    charSegmentation(imagesPath)
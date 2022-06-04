import os
import sys

import cv2
import numpy as np

from tqdm import tqdm

from utils import *
from pathlib import Path


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
    # print(box1, box2)
    holder = tuple(holder)
    return holder


def checkMerge(box1, box2):
    # Following lines merge two bounding boxes, if box1 is inside box2.
    # A leeway is given, so box1 can be 70% of its width outside box2 (on the x axis)
    if ((box2[0] - box1[2] * 7 / 10) <= box1[0]) and \
            ((box2[0] + box2[2] + box1[2] * 7 / 10) >= (box1[0] + box1[2])):
        # Following if statement is for the y axis. On this axis, if at least one of the edges of the current
        # bounding box is inside the previous bounding box the boxes will be merged.
        if (box2[1] < (box1[1] + box1[3]) <= (box2[1] + box2[3])) or \
                ((box2[1] + box2[3]) > box1[1] >= box2[1]):
            box1 = mergeBoxes(box1, box2)
            return box1
    else:
        return


def splitConnectedElements(image, box):
    focus = 255 - image[(box[1]):(box[1] + box[3] + 2), (box[0]):(box[0] + box[2] + 2)]
    copy = focus.copy()
    focus[focus <= 127] = 1
    focus[focus > 127] = 0
    projection = np.sum(focus, axis=0)
    if (((box[2]/box[3]) % 1) >= .35) and ((box[2]/box[3]) < 2):
        ratio = int(box[2]/box[3]) + 1
    elif ((box[2]/box[3]) % 1) >= .23:
        ratio = int(box[2] / box[3]) + 1
    else:
        ratio = int(box[2] / box[3])
    #print(box[2]/box[3], ratio)
    if ratio == 2:
        localMinimum = min(projection[(int(box[2] * 0.25)):(int(box[2] * 0.75))])
        splitPoint = (int(box[2] * 0.25)) +\
                     np.where(projection[(int(box[2] * 0.25)):(int(box[2] * 0.75))] == localMinimum)[0][0]
        cv2.line(copy, ((splitPoint - 2), 0), ((splitPoint - 2), focus.shape[0]),
                 color=(0, 255, 0),
                 thickness=6)
    elif ratio == 3:
        splitPoint = box[2]
        step = int(box[2] / ratio)
        for i in range(1, ratio):
            lowerMargin = splitPoint - step - int(box[2] * 0.15)
            if lowerMargin < 0:
                lowerMargin = 4
            upperMargin = splitPoint - step + int(box[2] * 0.15)
            localMinimum = min(projection[lowerMargin:upperMargin])
            splitPoint = lowerMargin + np.where(projection[lowerMargin:upperMargin] == localMinimum)[0][0]
            cv2.line(copy, ((splitPoint - 2), 0), ((splitPoint - 2), focus.shape[0]),
                     color=(0, 255, 0),
                     thickness=6)
    elif ratio == 4:
        splitPoint = box[2]
        step = int(box[2] / ratio)
        for i in range(1, ratio):
            lowerMargin = splitPoint - step - int(box[2] * 0.15)
            if lowerMargin < 0:
                lowerMargin = 4
            upperMargin = splitPoint - step + int(box[2] * 0.15)
            localMinimum = min(projection[lowerMargin:upperMargin])
            splitPoint = lowerMargin + np.where(projection[lowerMargin:upperMargin] == localMinimum)[0][0]
            cv2.line(copy, ((splitPoint - 2), 0), ((splitPoint - 2), focus.shape[0]),
                     color=(0, 255, 0),
                     thickness=6)
    elif ratio == 5:
        splitPoint = box[2]
        step = int(box[2] / ratio)
        for i in range(1, ratio):
            lowerMargin = splitPoint - step - int(box[2] * 0.10)
            if lowerMargin < 0:
                lowerMargin = 4
            upperMargin = splitPoint - step + int(box[2] * 0.10)
            localMinimum = min(projection[lowerMargin:upperMargin])
            splitPoint = lowerMargin + np.where(projection[lowerMargin:upperMargin] == localMinimum)[0][0]
            cv2.line(copy, ((splitPoint - 2), 0), ((splitPoint - 2), focus.shape[0]),
                     color=(0, 255, 0),
                     thickness=6)
        """
        posMin = min(projection[(int(box[2] * 0.7)):(int(box[2] - box[2] * 0.15))])
        for i in range((len(projection) - 3), int(len(projection) * 0.15), -1):
            if (projection[i] <= posMin) and (check >= int(len(projection) * 0.20)):
                cv2.line(copy, ((i - 2), 0), ((i - 2), focus.shape[0]),
                         color=(0, 255, 0),
                         thickness=6)
                if projection[i - 1] != projection[i]:
                    posMin = min(projection[(i - int(len(projection) * 0.15)):(i - 1)])
                    # print(posMin)
                    check = 0
            check += 1"""
    #print(projection, focus.shape, posMin)


    if not os.path.exists("lines/connected"):
        os.makedirs("lines/connected")
    cv2.imwrite("lines/connected/" + str(box[0]) + "--" + str(box[2])+ "--"+ str(box[3]) + ".png", copy)
    return


def cleanBoxes(image, box, bbox):
    # Bounding box is saved only if it is 15 pixels or bigger
    # This eliminates small bounding boxes on noise
    if (box[2] < 20) or (box[3] < 20):
        return
    # 1 character is somewhere between 25-40 pixels and 75 pixels in width
    # These characters are directly returned
    elif (box[2] > 40) and (box[2] < 95):
        return box
    # Taking care of connected components
    # TODO: find a way to split components
    elif box[2] > 94:
        print(box)
        splitConnectedElements(image, box)
        return box
    else:
        if len(bbox) >= 2:
            hold = checkMerge(bbox[-2], box)
            if hold:
                bbox.pop()
                bbox.append(hold)
                # print(hold)

        if len(bbox) >= 1:
            hold = checkMerge(box, bbox[-1])
            if hold:
                bbox.pop()
                bbox.append(hold)
                # print(hold)
            hold = checkMerge(bbox[-1], box)
            if hold:
                bbox.pop()
                bbox.append(hold)
                # print(hold)
        # Following lines merge two bounding boxes, if the current bounding box is inside the previous registered
        # bounding box. A leeway is given, so the current bounding bounding box can be 70% of its width outside the
        # previous bounding box (on the x axis)
        """if bbox and ((bbox[-1][0] - box[2]*7/10) <= box[0]) and\
                ((bbox[-1][0] + bbox[-1][2] + box[2]*7/10) >= (box[0] + box[2])):
            # Following if statement is for the y axis. On this axis, if at least one of the edges of the current
            # bounding box is inside the previous bounding box the boxes will be merged.
            if (bbox[-1][1] < (box[1] + box[3]) <= (bbox[-1][1] + bbox[-1][3])) or\
                    ((bbox[-1][1] + bbox[-1][3]) > box[1] >= bbox[-1][1]):
                box = mergeBoxes(bbox[-1], box)
                bbox.pop()
                print(box)
        # Following elif statement is the opposite of the last if statement. it does the same merging, but it checks if
        # the previous bounding box is inside the current one
        elif bbox and ((box[0] - bbox[-1][2]*7/10) <= bbox[-1][0]) and\
                ((box[0] + box[2] + bbox[-1][2]*7/10) >= (bbox[-1][0] + bbox[-1][2])):
            if (box[1] < (bbox[-1][1] + bbox[-1][3]) <= (box[1] + box[3])) or \
                    ((box[1] + box[3]) > bbox[1] >= box[1]):
                box = mergeBoxes(bbox[-1], box)
                bbox.pop()
                print(box)"""
        return box


def getBBox(image):
    # Getting characters' contours and creating a hierarchy
    cnt, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    poly = [None] * len(cnt)
    # print(hierarchy, "\n", cnt)

    # Declaring bounding box list
    bbox = []

    for i, contour in enumerate(cnt):
        poly[i] = cv2.approxPolyDP(contour, 3, closed=True)

        if hierarchy[0][i][3] == -1:
            box = cv2.boundingRect(poly[i])
            box = cleanBoxes(image, box, bbox)
            if box:
                ##print(box)
                bbox.append(box)

    return bbox


def oneLineProcessing(img_str, line):
    linePath = "lines/" + str(img_str) + "/" + str(line)
    # print(linePath)
    img = cv2.imread(linePath)
    processed = preprocess(img)

    # Getting the bounding boxes from the inverted version of the processed image
    bbox = getBBox(255 - processed)

    for i in range(len(bbox)):
        corner1, corner2, corner3, corner4 = int(bbox[i][0]), int(bbox[i][1]), int(bbox[i][2]), int(bbox[i][3])
        # print(corner1, corner2, corner3, corner4)
        color = list(np.random.random(size=3) * 256)
        cv2.rectangle(img, (4 * corner1, 4 * corner2),
                      (4 * (corner1 + corner3), 4 * (corner2 + corner4)),
                      color, shift=2, thickness=6)

    # cv2.imwrite("lines/x/3.jpg", img)
    # cv2.imwrite("lines/x/4.jpg", 255 - processed)
    if not os.path.exists("lines/bbox"):
        os.makedirs("lines/bbox")
    cv2.imwrite("lines/bbox/" + str(img_str[:-4]) + "--" + str(line), img)


def charSegmentation(imagesPath):
    for img in tqdm(imagesPath):
        print(img)
        if img != ".DS_Store":
            imgPath = os.listdir("lines/" + str(img) + "/")
            for line in imgPath:
                if (str(img) == "x") or (str(img) == "connected"):
                    break
                elif str(img) == "bbox":
                    break
                oneLineProcessing(img_str=img, line=line)


if __name__ == "__main__":
    imagesPath = os.listdir("lines/")
    charSegmentation(imagesPath)

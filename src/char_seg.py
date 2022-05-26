import os
import sys

import cv2
import numpy as np

from utils import *
from pathlib import Path


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
            bbox.append(box)

    return bbox


def oneLineProcessing(linePath):
    img = cv2.imread(linePath)
    processed = preprocess(img)

    # Getting the bounding boxes from the inverted version of the processed image
    bbox = getBBox(255 - processed)

    for i in range(len(bbox)):
        corner1, corner2, corner3, corner4 = int(bbox[i][0]), int(bbox[i][1]), int(bbox[i][2]), int(bbox[i][3])
        # print(corner1, corner2, corner3, corner4)
        cv2.rectangle(img, (4*corner1, 4*corner2),
                      (4*(corner1+corner3), 4*(corner2+corner4)),
                      (0, 255, 0), shift=2, thickness=6)

    cv2.imwrite("lines/x/3.jpg", img)
    cv2.imwrite("lines/x/4.jpg", 255 - processed)


if __name__ == "__main__":
    imagesPath = os.listdir("lines/")
    linesImg0 = os.listdir("lines/" + imagesPath[0] + "/")
    linePath = "lines/" + str(imagesPath[0]) + "/" + str(linesImg0[0])
    print(linePath)
    oneLineProcessing(linePath=linePath)

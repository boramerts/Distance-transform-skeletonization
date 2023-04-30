#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 12:13:36 2023

@author: boramert
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove
import os


def getSkeleton(datapath, file):
    image_raw = Image.open(datapath)
    
    image_raw = np.array(image_raw)

    plt.imshow(image_raw, cmap='gray')
    plt.title("Original Image")
    plt.show()

    # AI background removal
    image = remove(image_raw)

    bg = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY) - \
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # type: ignore

    mask = bg < 10
    bg = bg*0
    bg[mask] = 255

    # Distance Transform
    _, threshold = cv2.threshold(bg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Calculate the distance transform
    distTransform = cv2.distanceTransform(threshold, cv2.DIST_L2, 3)

    plt.imshow(distTransform,cmap='gray')
    plt.title("Distance Transform")
    plt.show()

    kernel = np.ones((5, 5), np.float32)/25
    distTransform = cv2.filter2D(distTransform, -1, kernel)

    # Local Maximum Points
    lmp_kernel = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])
    LMP = distTransform*0
    for row in range(1, len(distTransform)-2):
        for col in range(1, len(distTransform[0])-2):
            for k_row in range(0, 5):
                for k_col in range(0, 5):
                    lmp_kernel[k_row, k_col] = distTransform[row -
                                                             (2-k_row), col-(2-k_col)]
            if (lmp_kernel[2, 2] == np.max(lmp_kernel)) and (lmp_kernel[2, 2] > 0):
                LMP[row, col] = 255

    x_grad = np.gradient(distTransform)

    plt.imshow(x_grad[0])
    plt.title("df/dx")
    plt.show()

    plt.imshow(x_grad[1])
    plt.title("df/dy")
    plt.show()

    delta_dt = np.sqrt(np.square(x_grad[0])+np.square(x_grad[1]))

    plt.imshow(delta_dt)
    plt.title("Gradient of Distance Transform")
    plt.show()

    # Critical Points
    CP = LMP*0
    cp_kernel = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]])

    for row in range(1, len(delta_dt)-1):
        for col in range(1, len(delta_dt[0])-1):
            for k_row in range(0, 3):
                for k_col in range(0, 3):
                    cp_kernel[k_row, k_col] = delta_dt[row -
                                                       (1-k_row), col-(1-k_col)]
            if (cp_kernel[1, 1] == np.min(cp_kernel)) and (LMP[row, col] == 255):
                CP[row, col] = 255

    plt.imshow(CP)
    plt.title("Critical Points")
    plt.show()

    if datapath.find("downdog") != -1:
        path = '/Users/boramert/Desktop/Okul/BİL 587/Proje/DistanceTransform/downdog/' + file
        cv2.imwrite(path, CP)
    elif datapath.find("goddess") != -1:
        path = '/Users/boramert/Desktop/Okul/BİL 587/Proje/DistanceTransform/goddess/' + file
        cv2.imwrite(path, CP)
    elif datapath.find("plank") != -1:
        path = '/Users/boramert/Desktop/Okul/BİL 587/Proje/DistanceTransform/plank/' + file
        cv2.imwrite(path, CP)
    elif datapath.find("tree") != -1:
        path = '/Users/boramert/Desktop/Okul/BİL 587/Proje/DistanceTransform/tree/' + file
        cv2.imwrite(path, CP)
    elif datapath.find("warrior2") != -1:
        path = '/Users/boramert/Desktop/Okul/BİL 587/Proje/DistanceTransform/warrior2/' + file
        cv2.imwrite(path, CP)


rootdir = "/Users/boramert/Desktop/Okul/BİL 587/Proje/DATASET"

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        datapath = subdir+"/"+file
        if datapath.find(".jpg") != -1 and datapath.find(".icloud") == -1:
            print("Processing.......")
            print(datapath)
            getSkeleton(datapath, file)
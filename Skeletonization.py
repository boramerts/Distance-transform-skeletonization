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

image_raw = Image.open("/Users/boramert/Desktop/Okul/BİL 587/Proje/DATASET/TRAIN/goddess/00000097.jpg")
#image_raw = Image.open("/Users/boramert/Desktop/Okul/BİL 587/Proje/sqq.jpeg")

plt.imshow(image_raw)
plt.title("raw image")
plt.show()

image_raw = np.array(image_raw)

#AI background removal
image = remove(image_raw)

plt.imshow(image)
plt.title("bg removed image")
plt.show()

plt.imshow(cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)-cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
plt.title("bg image")
plt.show()

bg = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)-cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

mask = bg<10
bg = bg*0
bg[mask] = 255

plt.imshow(bg)
plt.title("body image")
plt.show()

#Distance Transform
#_, threshold = cv2.threshold(image_gray, 123, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
_, threshold = cv2.threshold(bg, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.imshow(threshold)
plt.title("threshold")
plt.show()

# Calculate the distance transform
# distTransform = cv2.distanceTransform(threshold, cv2.DIST_C, 3)
distTransform= cv2.distanceTransform(threshold, cv2.DIST_L2, 3)
#distTransform= cv2.distanceTransform(threshold, cv2.DIST_C, 3)

#distTransform = cv2.GaussianBlur(distTransform,(5,5),0)
#distTransform = cv2.blur(distTransform, (5,5))

kernel = np.ones((5,5),np.float32)/25
distTransform = cv2.filter2D(distTransform,-1,kernel)

# Display the distance image
plt.imshow(distTransform)
plt.title("Distance Transform ")
plt.show()

#Local Maximum Points
lmp_kernel = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])
LMP = distTransform*0
for row in range(1,len(distTransform)-2):
      for col in range(1,len(distTransform[0])-2):
          for k_row in range(0,5):
              for k_col in range(0,5):
                  lmp_kernel[k_row,k_col] = distTransform[row-(2-k_row),col-(2-k_col)]
          if (lmp_kernel[2,2] == np.max(lmp_kernel)) and (lmp_kernel[2,2] > 0):
              LMP[row,col] = 255

plt.imshow(LMP)
plt.title("LMP")
plt.show()

x_grad = np.gradient(distTransform)

plt.imshow(x_grad[0])
plt.title("x_grad 0")
plt.show()

plt.imshow(x_grad[1])
plt.title("x_grad 1")
plt.show()

delta_dt = np.sqrt(np.square(x_grad[0])+np.square(x_grad[1]))

plt.imshow(delta_dt)
plt.title("delta_dt")
plt.show()

#distTransform = cv2.normalize(distTransform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

CP = LMP*0
cp_kernel = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])

for row in range(1,len(delta_dt)-1):
      for col in range(1,len(delta_dt[0])-1):
          for k_row in range(0,3):
              for k_col in range(0,3):
                  cp_kernel[k_row,k_col] = delta_dt[row-(1-k_row),col-(1-k_col)]
          if (cp_kernel[1,1] == np.min(cp_kernel)) and (LMP[row,col] == 255):
              CP[row,col] = 255

plt.imshow(CP)
plt.title("Critical Points")
plt.show()

CP = cv2.normalize(CP, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

cnts = cv2.findContours(CP , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

cimg = image_raw*0

#cv2.drawContours(cimg, cnts, -1, (255,255,255), 1)

# plt.imshow(cimg)
# plt.title("Contours")
# plt.show()

for i in range(len(cnts)):
    min_dist = max(image_raw.shape[0], image_raw.shape[1])
    cl = []
    
    ci = cnts[i]
    ci_left = tuple(ci[ci[:, :, 0].argmin()][0])
    ci_right = tuple(ci[ci[:, :, 0].argmax()][0])
    ci_top = tuple(ci[ci[:, :, 1].argmin()][0])
    ci_bottom = tuple(ci[ci[:, :, 1].argmax()][0])
    ci_list = [ci_bottom, ci_left, ci_right, ci_top]
    
    for j in range(i + 1, len(cnts)):
        cj = cnts[j]
        cj_left = tuple(cj[cj[:, :, 0].argmin()][0])
        cj_right = tuple(cj[cj[:, :, 0].argmax()][0])
        cj_top = tuple(cj[cj[:, :, 1].argmin()][0])
        cj_bottom = tuple(cj[cj[:, :, 1].argmax()][0])
        cj_list = [cj_bottom, cj_left, cj_right, cj_top]
        
        for pt1 in ci_list:
            for pt2 in cj_list:
                dist = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))     #dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
                if dist < min_dist:
                    min_dist = dist             
                    cl = []
                    cl.append([pt1, pt2, min_dist])
    if len(cl) > 0:
        cv2.line(cimg, cl[0][0], cl[0][1], (255, 255, 255), thickness = 1)

plt.imshow(cimg)
plt.title("Image with Borders")
plt.show()

# # Save the transformed image
# cv2.imwrite('CriticalPoints.png', CP)
  
# apply the cv2.cornerHarris method
# to detect the corners with appropriate
# values as input parameters
dest = cv2.cornerHarris(CP, 2, 5, 0.07)
  
# Results are marked through the dilated corners
dest = cv2.dilate(dest, None)
  
# Reverting back to the original image,
# with optimal threshold value
image_raw[dest > 0.01 * dest.max()]=[0, 0, 255]
  
plt.imshow(image_raw)
plt.title("Image with Borders")
plt.show()






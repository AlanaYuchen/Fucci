#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 17:04:27 2020

@author: jefft
"""
import os
os.chdir('/Users/jefft/Desktop/BMI_Project/Fucci')

# Adaptative Region Grow Mechanism
import numpy as np
import skimage.io as io 
 
class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
 
    def getX(self):
        return self.x
    def getY(self):
        return self.y
 
def getGrayDiff(img,currentPoint,tmpPoint):
    return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))
 
def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [ Point(0, -1),  Point(1, 0),Point(0, 1), Point(-1, 0)]
    return connects
 
def regionGrow(img,seeds,thresh,p = 1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    out = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
        if seedMark[currentPoint.x, currentPoint.y] == 0:
            # if not visited
            seedMark[currentPoint.x,currentPoint.y] = label
            for i in range(8):
                tmpX = currentPoint.x + connects[i].x
                tmpY = currentPoint.y + connects[i].y
                if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                    continue
                grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
                if grayDiff < thresh and seedMark[tmpX,tmpY] == 0: # visited point not included
                    seedMark[tmpX,tmpY] = label # mark as visited
                    if img[tmpX, tmpY] > 30:
                        out[tmpX,tmpY] = label
                    seedList.append(Point(tmpX,tmpY))
    return out
 
def Sobel_segmentation(image):
    # input: uint8 / float8 image
   Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
   Gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

   nrows = image.shape[0]
   ncols = image.shape[1]
   mag = np.zeros((nrows, ncols))

   for i in range(nrows-2):
       for j in range(ncols-2):
           factor = image[i+1, j+1] + 1
           S1 = np.sum(np.multiply(Gx,image[i:i+3, j:j+3])) * 256 / factor
           S2 = np.sum(np.multiply(Gy,image[i:i+3, j:j+3])) * 256 / factor
           mag[i+1,j+1] = np.sqrt(pow(S1,2)+pow(S2,2))
   
   return mag

img = io.imread('data/test_seg.png')
seed_crd = np.where(img>np.percentile(img, 96))
seeds = list(map(lambda x: Point(seed_crd[0][x], seed_crd[1][x]), range(len(seed_crd[1]))))
# seeds = [Point(540,570),Point(842,242),Point(634,562)]
binaryImg = regionGrow(img,seeds,2)
io.imshow(binaryImg)

import skimage.filters as filters
img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
img = img.astype('uint8')
img_sobel_man = Sobel_segmentation(img)
img_sobel = filters.sobel(img)
io.imshow(img_sobel)
io.imsave('/Users/jefft/Desktop/test_sobel.png', img_sobel)
io.imshow(img_sobel_man)
io.imsave('/Users/jefft/Desktop/test.png',img_sobel_man)

import skimage.measure as measure
frame = measure.regionprops(measure.label(img_sobel_man),intensity_image = img_sobel_man)



dt = io.imread('/Users/jefft/Desktop/BMI_Project/Fucci/data/P1_ave_gau.tif')
out = np.empty(dt.shape)
for i in range(20):
    out[i,:,:] = Sobel_segmentation(dt[i,:,:])
    print(i)

io.imsave('/Users/jefft/Desktop/test.tif', out[0:20,:,:])



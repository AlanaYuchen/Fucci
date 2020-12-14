#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 09:34:31 2020

@author: jefft
"""
from skimage.draw import line
import cv2
from numpy import argsort

def doGmSeg(image, trh=1000):
    # Input: binary image containing clustered masks
    # Output: separated cluster
    
    ctrs = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    
    for i in range(len(ctrs)):
        hull = cv2.convexHull(ctrs[i], returnPoints = False) # return indices of con
        defects = cv2.convexityDefects(ctrs[i], hull) # shape: number of defect points * 1 * four params: start_index, end_index, fartherest point index and distance. 
        # filter defects by threshold, defalut 1000
        defects = defects[defects[:,:,3]>trh]
        if (defects.shape[0]<2):
            # if no over threshold convexity defect found, return early
            # Note: if there is only one defect point, no segmentation will be performed.
            # TODO > 2 cases
            continue
        else:
            # if more than 2, take maximum 2
            defects = defects[argsort(-defects[:,3].ravel())[0:2]] # descending sort by depth
            # extract candidate points along the contour
            this_ctrs = ctrs[i][:,0,:]
            cd = this_ctrs[defects[:,2].ravel(),:]
            rr, cc = line(cd[0,1],cd[0,0],cd[1,1],cd[1,0])
            image[rr, cc] = 1
            image[rr+1, cc] = 1
            image[rr, cc+1] = 1
    
    return image

#================================ Testing =====================================
'''
import skimage.io as io
img = io.imread('data/gmSeg_test.tif')
io.imsave('data/gmSeg_test_out.tif',doGmSeg(img, trh=1000))
'''

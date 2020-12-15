#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 22:08:13 2020

@author: Full Moon
"""

import math
import numpy as np
import time
import skimage.io as io
import skimage.exposure as exposure
import skimage.filters as filters
import skimage.measure as measure
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening
from skimage.draw import line
import cv2
from numpy import argsort
from scipy import ndimage as ndi

def doGmSeg(image, trh=1000):
    # Input: binary image containing clustered masks
    # Output: separated cluster
    
    ctrs = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    for i in range(len(ctrs)):
        hull = cv2.convexHull(ctrs[i], returnPoints = False) # return indices of con
        defects = cv2.convexityDefects(ctrs[i], hull) # shape: number of defect points * 1 * four params: start_index, end_index, fartherest point index and distance. 
        if not defects is None:
        # filter defects by threshold, defalut 1000
            defects = defects[defects[:,:,3]>trh]
            if (defects.shape[0]<2):
                # if no over threshold convexity defect found, return early
                # Note: if there is only one defect point, no segmentation will be performed.
                continue
            else:
                # if more than 2, take maximum 2
                defects = defects[argsort(-defects[:,3].ravel())[0:2]] # descending sort by depth
                # extract candidate points along the contour
                this_ctrs = ctrs[i][:,0,:]
                cd = this_ctrs[defects[:,2].ravel(),:]
                rr, cc = line(cd[0,1],cd[0,0],cd[1,1],cd[1,0])
                image[rr, cc] = 0
                image[rr+1, cc] = 0
                image[rr, cc+1] = 0
    
    return image

def adaptive_rescaling(pcd, k=5):
    # pcd: Sobel processed image, float 64
    p_min = np.min(pcd)
    p_max = np.max(pcd)
    for i in range(pcd.shape[0]):
        for j in range(pcd.shape[1]):
            pcd[i,j] = math.log(k * (pcd[i,j]-p_min) / (p_max - p_min) + 1) * 255 / math.log(k + 1)
    
    return pcd.astype('uint8')

def doSeg(gfp_path, mcy_path):
    
    start = time.time()
    
    # Input: gfp and mCherry file path
    gfp=io.imread(gfp_path)
    mcy=io.imread(mcy_path)
    rng = gfp.shape[0]
    
    #rng = 30
    
    # Enhance contrast: histogram equalization
    gfp = np.asarray(list(map(lambda x: exposure.rescale_intensity(gfp[x,:,:], in_range=tuple(np.percentile(gfp[x,:,:], (2, 98)))),range(rng))))
    mcy = np.asarray(list(map(lambda x: exposure.rescale_intensity(mcy[x,:,:], in_range=tuple(np.percentile(mcy[x,:,:], (2, 98)))), range(rng))))
    
    gfp=gfp.astype('uint32')
    mcy=mcy.astype('uint32')
    # take average intensity
    ave=np.add(gfp,mcy)/2/65535*255
    
    #Gaussian filter
    ave_gau = np.asarray(list(map(lambda x: filters.gaussian(ave[x,:,:],sigma=2), range(rng))))

    # sobel kernal and adaptive intensity stretching
    ave_gau = np.asarray(list(map(lambda x: filters.sobel(ave_gau[x,:,:]), range(rng))))
    ave_gau = np.asarray(list(map(lambda x: adaptive_rescaling(ave_gau[x,:,:], k=15), range(rng))))
    
    end_sobel = time.time()
    print("Finished edge detection: " + str(math.floor(end_sobel-start)) + " s.")

    # global otsu
    otsu = np.asarray(list(map(lambda x: threshold_otsu(ave_gau[x,:,:]),range(rng))))
    mask = np.asarray(list(map(lambda x: ave_gau[x,:,:]>=otsu[x],range(rng))))
    # fill holes and filter small object
    for i in range(mask.shape[0]):
        mask[i,:,:] = binary_opening(mask[i,:,:])
        mask[i,:,:] = ndi.binary_fill_holes(mask[i,:,:])
        mask[i,:,:] = remove_small_objects(mask[i,:,:].astype('bool'), 1000) # size filter
    
    mask = mask.astype('uint8') * 255
    for i in range(mask.shape[0]):
        # convex hull filling
        props = measure.regionprops(measure.label(mask[i,:,:]), intensity_image=None)
    
        for k in range(len(props)):
            box = props[k].bbox
            if props[k].convex_area > 3 * props[k].area:
                mask[i,box[0]:box[2],box[1]:box[3]] = props[k].convex_image.copy() * 255

        # geometric segmentation
        mask[i,:,:] = doGmSeg(mask[i,:,:], trh=5000)

    end = time.time()
    print("Finished generating mask: " + str(math.floor(end-start)) + " s.")
    # Output: mask, intensity rescaled gfp and mCherry image stacks
    return mask, gfp, mcy
    
#============================= Testing ========================================
gfp_path = 'data/P1_gfp.tif'
mcy_path = 'data/P1_mCherry.tif'
out = doSeg(gfp_path,mcy_path)

'''
t = ave_gau[7,:,:].copy()
t_p_ori = threshold_otsu(t)
t_ori_pcd = t >= t_p_ori
t = adaptive_rescaling(t, k=7)
t_p = threshold_otsu(t)
t_pcd = t >= t_p

io.imshow(t_ori_pcd)
io.imshow(t_pcd)
io.imsave('data/adp_rscl_otsu.tif', t_pcd)

import tifffile as tif
tif.imsave('data/mask_test.tif', mask)
'''

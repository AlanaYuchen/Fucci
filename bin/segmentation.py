#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 22:08:13 2020

The script contains main segmentation function doSeg(gfp_path, mcy_path) that 
generate binary masks from raw fluroscent microscopy images. Major steps includes:
intensity rescaling (also output), channel combination via averaging, Sobel edge
detection, adaptive edge intensity rescaling, binarization, hole filling and 
naive geometric segmentation.

The script is able to recover most of the weak edges, though, naive geometric 
segmentation results in much over-segmentation.

Time performance is evaluated as ~ 600 s per 289 * 1200 * 1200 dataset (dimension: txy).

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
from skimage.morphology import remove_small_objects, binary_opening, binary_erosion
from scipy import ndimage as ndi
import skimage.segmentation as segmentation

def doWaterShed(image):
    image = image.astype('bool')
    distance = ndi.distance_transform_edt(image)
    distance = filters.gaussian(distance, sigma=10)
    mask_f = distance>=0.6*np.max(distance)
    markers = ndi.label(mask_f)[0]
    labels = segmentation.watershed(-distance, markers, mask=image, watershed_line=True)
    return labels

def adaptive_rescaling(pcd, k=5):
    # The function applies a non-linear function to each pixel of the image, 
    # which amplify low-intensity signals
    # Input: pcd: float 64 image; k: factor, bigger k results in greater enhancement
    # Output: uint8 image
    
    p_min = np.min(pcd)
    p_max = np.max(pcd)
    for i in range(pcd.shape[0]):
        for j in range(pcd.shape[1]):
            pcd[i,j] = math.log(k * (pcd[i,j]-p_min) / (p_max - p_min) + 1) * 255 / math.log(k + 1)
    
    return pcd.astype('uint8')

def doSeg(gfp_path, mcy_path):
    
    # This is the main function for segmentation, see notes at the beginning of the script
    
    start = time.time()
    
    # Input: gfp and mCherry file path
    gfp=io.imread(gfp_path)
    mcy=io.imread(mcy_path)
    rng = gfp.shape[0]
    
    # Enhance contrast: intensity rescaling
    gfp = np.asarray(list(map(lambda x: exposure.rescale_intensity(gfp[x,:,:], in_range=tuple(np.percentile(gfp[x,:,:], (2, 98)))),range(rng))))
    mcy = np.asarray(list(map(lambda x: exposure.rescale_intensity(mcy[x,:,:], in_range=tuple(np.percentile(mcy[x,:,:], (2, 98)))), range(rng))))
    
    gfp=gfp.astype('uint32')
    mcy=mcy.astype('uint32')
    # Take average intensity, suppress noise while enhancing transition signal
    ave=np.add(gfp,mcy)/2/65535*255
    
    #Gaussian filter, to denoise
    ave_gau = np.asarray(list(map(lambda x: filters.gaussian(ave[x,:,:],sigma=2), range(rng))))

    # Sobel kernal and adaptive intensity stretching
    ave_gau = np.asarray(list(map(lambda x: filters.sobel(ave_gau[x,:,:]), range(rng))))
    ave_gau = np.asarray(list(map(lambda x: adaptive_rescaling(ave_gau[x,:,:], k=15), range(rng))))
    
    end_sobel = time.time()
    print("Finished edge detection: " + str(math.floor(end_sobel-start)) + " s.")

    # global otsu thresholding
    otsu = np.asarray(list(map(lambda x: threshold_otsu(ave_gau[x,:,:]),range(rng))))
    mask = np.asarray(list(map(lambda x: ave_gau[x,:,:]>=otsu[x],range(rng))))

    for i in range(mask.shape[0]):
        mask[i,:,:] = binary_opening(mask[i,:,:]) # open calculation, to denoise
        mask[i,:,:] = ndi.binary_fill_holes(mask[i,:,:]) # fill holes
        mask[i,:,:] = remove_small_objects(mask[i,:,:].astype('bool'), 1000) # size filter
    
    mask = mask.astype('uint8') * 255
    for i in range(mask.shape[0]):
        # inspect convex hull to fill non-connected ring edges
        props = measure.regionprops(measure.label(mask[i,:,:]), intensity_image=None)
    
        for k in range(len(props)):
            box = props[k].bbox
            if props[k].convex_area > 3 * props[k].area:
                mask[i,box[0]:box[2],box[1]:box[3]] = props[k].convex_image.copy() * 255
            else:
                lbs = doWaterShed(props[k].image)
                if np.max(lbs)>1:
                    lbs[lbs>=1] = 255
                    mask[i,box[0]:box[2],box[1]:box[3]] = lbs.copy()
    
        # erosion, to exclude super thick edges
        mask[i,:,:] = binary_erosion(binary_erosion(mask[i,:,:]))
    
    mask =  mask.astype('uint8') * 255
    end = time.time()
    print("Finished generating mask: " + str(math.floor(end-start)) + " s.")
    
    # Output: mask, intensity rescaled gfp and mCherry image stacks
    return mask, gfp, mcy
    
#============================= Testing ========================================
'''
gfp_path = 'data/P1_gfp.tif'
mcy_path = 'data/P1_mCherry.tif'
out = doSeg(gfp_path,mcy_path)
'''
#mask = io.imread('../data/P1_mask_out.tif')
#image = mask[20,:,:]
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

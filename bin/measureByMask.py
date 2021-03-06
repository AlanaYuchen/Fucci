#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 22:53:26 2020

@author: Full Moon
"""

import numpy as np
import skimage.measure as measure
import skimage.io as io
import pandas as pd
import math
import skimage.transform as transform

def doMeasure(mask, gfp, mcy, dic_path):
    # mask: binary image, from segmentation
    # gfp: rescaled gfp from segmentation
    # mcy: rescaled mCherry from segmentation
    # dic: DIC image path
    
    # All input dimension should be (FRAME_NUMBER, 1200, 1200)
    
    # load DIC image
    dic = io.imread(dic_path)

    # Initialize table columns
    x = []
    y = []
    frame = []
    gfp_intensity = [] # intensity after intensity rescaled in segmentation step.
    mcy_intensity = []
    bbox = []
    stacks = [] # keep trace of images
    area = []
    frame_num = np.shape(mask)[0] # get time-lapse length
    mask = mask.astype('bool')
    
    for j in range(frame_num):
        # for each time frame
        label = measure.label(mask[j,:,:], connectivity=1)
        gfp_frame = measure.regionprops(label, intensity_image = gfp[j,:,:])
        mcy_frame = measure.regionprops(label, intensity_image = mcy[j,:,:]) # region property generator
        
        for i in range(len(gfp_frame)):
            # For each object in the time frame, record properties.
            # A secondary size filter excludes small particles after geometric segmentation.
           
            if gfp_frame[i].area > 1000:
               
                gfp_intensity.append(gfp_frame[i].mean_intensity)
                mcy_intensity.append(mcy_frame[i].mean_intensity)
                area.append(gfp_frame[i].area)
                
                # Construct object image with all three channels, resized for classification.
                bbox_obj = gfp_frame[i].bbox # object bounding box
                bbox.append(bbox_obj)
                # masking three channels by multiplying region object mask
                gfp_obj = np.multiply(gfp[j, bbox_obj[0]:bbox_obj[2], bbox_obj[1]:bbox_obj[3]], gfp_frame[i].image)
                mcy_obj = np.multiply(mcy[j, bbox_obj[0]:bbox_obj[2], bbox_obj[1]:bbox_obj[3]], gfp_frame[i].image)
                dic_obj = np.multiply(dic[j, bbox_obj[0]:bbox_obj[2], bbox_obj[1]:bbox_obj[3]], gfp_frame[i].image)
                stack = np.stack([dic_obj, gfp_obj, mcy_obj], axis=2)
                stack_resized = transform.resize(stack, (80,80,3)) # resize
                stack_resized = (stack_resized-np.min(stack_resized)) / (np.max(stack_resized) - np.min(stack_resized))
                stacks.append(stack_resized.astype('float32'))
            
                frame.append(j)
                x.append(math.ceil(gfp_frame[i].centroid[0])) # round centroids, as primary keys.
                y.append(math.ceil(gfp_frame[i].centroid[1]))

    dt = pd.DataFrame({"id":[x for x in range(1,len(x)+1)],"x":x,"y":y,"frame":frame,"gfp_intensity":gfp_intensity,"mcy_intensity":mcy_intensity,
                      "bbox":bbox,"area":area})
    return dt, stacks

#=============================== Testing ======================================
'''
import os
import skimage.io as io
os.chdir(os.path.abspath("../")) # project path, absolute
# Test constants
GFP_path = 'data/gfp_rscl.tif'
mCherry_path = 'data/mcy_rscl.tif'
MASK_path = 'data/P1_mask.tif'
DIC_path = 'data/P1_DIC.tif'

mask = io.imread(MASK_path)
gfp = io.imread(GFP_path)
mcy = io.imread(mCherry_path)
out = doMeasure(mask, gfp, mcy, DIC_path)
'''


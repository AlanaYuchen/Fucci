#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 22:53:26 2020

@author: Full Moon
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.measure as measure
import skimage.io as io
import pandas as pd
import math
import skimage.transform as transform

os.chdir(os.path.abspath("../")) # project path, absolute
# Constants
GFP_path = 'data/gfp_rscl.tif'
mCherry_path = 'data/mcy_rscl.tif'
MASK_path = 'data/P1_Mask.tif'
DIC_path = 'data/P1_DIC.tif'

# All input dimension should be (FRAME_NUMBER, 1200, 1200)
mask = io.imread(MASK_path)
gfp = io.imread(GFP_path)
mcy = io.imread(mCherry_path)
dic = io.imread(DIC_path)
print(np.shape(mask), np.shape(gfp), np.shape(mcy)) # check dimention
frame_num = np.shape(mask)[0] # get time-lapse length

# Measure intensity of all objects in dataset
# Reference: https://blog.csdn.net/u013066730/article/details/87971770
x = []
y = []
frame = []
gfp_intensity = []
mcy_intensity = []
bbox = []
stacks = []
area = []
for j in range(frame_num):
    # for each time frame
    gfp_frame = measure.regionprops(
        measure.label(mask[j,:,:]),
        intensity_image = gfp[j,:,:])
    mcy_frame = measure.regionprops(
        measure.label(mask[j,:,:]),
        intensity_image = mcy[j,:,:])
    for i in range(len(gfp_frame)):
        mask_current = np.zeros((1200,1200))
        if gfp_frame[i].area > 1000:
            # initial size filter: more than 1000 excluded.
            # for each object in the time frame
            gfp_intensity.append(gfp_frame[i].mean_intensity)
            mcy_intensity.append(mcy_frame[i].mean_intensity)
            area.append(gfp_frame[i].area)
            
            bbox_obj = gfp_frame[i].bbox
            bbox.append(bbox_obj)
            gfp_obj = np.multiply(gfp[j, bbox_obj[0]:bbox_obj[2], bbox_obj[1]:bbox_obj[3]], gfp_frame[i].image)
            mcy_obj = np.multiply(mcy[j, bbox_obj[0]:bbox_obj[2], bbox_obj[1]:bbox_obj[3]], gfp_frame[i].image)
            dic_obj = np.multiply(dic[j, bbox_obj[0]:bbox_obj[2], bbox_obj[1]:bbox_obj[3]], gfp_frame[i].image)
            stack = np.stack([dic_obj, gfp_obj, mcy_obj], axis=2)
            stack_resized = transform.resize(stack, (80,80,3)) * 255
            stack_resized = stack_resized.astype('uint8')
            stacks.append(stack_resized)
            
            frame.append(j)
            x.append(math.ceil(gfp_frame[i].centroid[0]))
            y.append(math.ceil(gfp_frame[i].centroid[1]))
            
dt = pd.DataFrame({"id":[x for x in range(1,len(x)+1)],"x":x,"y":y,"frame":frame,"gfp_intensity":gfp_intensity,"mcy_intensity":mcy_intensity,
                  "bbox":bbox,"area":area, "imageResized":stacks})


# ***optional***
# Training set reference table, no use when performing analysis.
ref = pd.read_csv('data/training/cls.csv')
ref['Center of the object_0'] = list(map(math.ceil, list(ref['Center of the object_0']))) # correct coordinate difference
ref['Center of the object_1'] = list(map(math.ceil, list(ref['Center of the object_1'])))
ref.rename(columns={'Center of the object_1':'x', 'Center of the object_0':'y'},inplace=True)
ref = ref[['frame','User Label','x','y']] # select useful columns
dt_label = pd.merge(dt, ref, on=('frame','x','y'))
dt_label = dt_label.sort_values(by="id")

# ***optional***
# Output training set
valid_num = 150 # number of validation set
valid = np.random.choice(range(dt_label.shape[0]), valid_num)
for i in range(dt_label.shape[0]):
    id = dt_label.iloc[i,0] # id
    label = dt_label.iloc[i,9] # user label
    image = dt_label.iloc[i,8] # image
    if i in valid:
        io.imsave('data/training/valid/valid_'+str(id)+"_"+label+".tif", image)
    else:
        io.imsave('data/training/train/train_'+str(id)+"_"+label+".tif", image)

'''
=========================== BELOW NOT TESTED =================================
'''
# ***optional***
# Store training metadata
dt_label[['id','frame','User Label','x','y','area','gfp_intensity','mcy_intensity']].to_csv('data/intensity.csv')


# ***optional***
# Join table with tracking result
track = pd.read_csv('data/P1_mask-data_CSV-Table.csv') # ilastik tracking output, coordinate difference = 1
track['Center_of_the_object_0'] += 1 # correct coordinate difference
track['Center_of_the_object_1'] += 1
track.rename(columns={'Center_of_the_object_1':'x', 'Center_of_the_object_0':'y'},inplace=True) # rename for inner join
track = track[['frame','trackId','x','y']] # select useful columns
dt = pd.merge(dt, track, on=('frame','x','y'))

trackId = 2
dt_filtered = dt[dt['trackId']==trackId]
plt.scatter(dt_filtered['mcy_intensity'], dt_filtered['gfp_intensity'], marker=".", color='red')
plt.plot(dt_filtered['mcy_intensity'], dt_filtered['gfp_intensity'], linestyle="dashed", color='blue')
plt.title('Stage 1, track ' + str(trackId), fontsize=24)
plt.xlabel('mCherry', fontsize=14)
plt.ylabel('GFP', fontsize=14)
'''
# bug, add frame label to each point
for i in range(dt_filtered.shape[0]-1):
    plt.text(x=dt_filtered['mcy_intensity'][i], y=dt_filtered['gfp_intensity'][i], s=str(dt_filtered['frame'][i]),
            fontsize=10, color = "r", verticalalignment='center', horizontalalignment='right')
'''
plt.show()




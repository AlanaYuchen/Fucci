#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 14:04:46 2020

@author: jefft
"""
import os
os.chdir('/Users/jefft/Desktop/BMI_Project/Fucci')

from Tracking import CentroidTracker
import pandas as pd

# load  metadata
meta = pd.read_csv('data/intensity.csv')

ct = CentroidTracker(maxDisappeared=10)
trackId = []
for i in range(max(meta['frame'])+1):
    # for each frame, extract centroids
    cur_frame = meta[meta['frame']==i].copy()
    if cur_frame.shape[0]==0:
        print("Warning! Frame with no object AT: " + str(i))
        continue
    centroids = list(map(lambda x: (list(cur_frame['x'])[x], list(cur_frame['y'])[x]), range(cur_frame.shape[0])))
    # do tracking
    objects = ct.update(centroids)
    # match track id back to centroids
    objects = {v: k for k, v in objects.items()}
    for c in centroids:
        if c in objects.keys():
            trackId.append(objects[c])
        else:
            trackId.append(-1) # not tracked
        
meta['track_id'] = trackId
meta['parent_track'] = [-1 for _ in range(len(trackId))] # inatialize parent track for mitosis prediction


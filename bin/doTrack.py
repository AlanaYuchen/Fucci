#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 14:04:46 2020

@author: Full Moon
"""

from tracking import CentroidTracker
import trackRefinepy
import pandas as pd

def centroidTracking(meta):
    ct = CentroidTracker(maxDisappeared=1, dist_trh=90)
    trackId = []
    for i in range(max(meta['frame'])+1):
        # for each frame, extract centroids
        cur_frame = meta[meta['frame']==i].copy()
        if cur_frame.shape[0]==0:
            print("Warning! Frame with no object detected AT: " + str(i))
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
            
    meta['trackId'] = trackId
    meta['lineageId'] = trackId
    meta['parentTrackId'] = [-1 for _ in range(len(trackId))] # inatialize parent track for mitosis prediction

    return trackRefinepy.doTrackRefine(meta)

# =============================== Testing =====================================
'''
import os
import pandas as pd
os.chdir(os.path.abspath("../")) # project path, absolute
# load test data
meta_data = pd.read_csv('data/intensity.csv')
'''
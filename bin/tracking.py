#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:56:37 2020

@author: Full Moon
"""

# Centroid tracking, by evaluating cross-frame Euclidean distance between bounding boxes

# Ref: https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
class CentroidTracker():
    def __init__(self, maxDisappeared=5):
        self.nextObjectID = 0 # counter of the current object number, for next object assignment. 
        self.objects = OrderedDict() # key: object ID; value: centroid
        self.disappeared = OrderedDict() # key: object ID; value: times of "disappear" mark.
      		# if cannot match for frame number larger than maxDisappeared, deregister from the list.
        self.maxDisappeared = maxDisappeared
    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
    
    def update(self, centroids):
        # Input: rects: bounding box collection, in format of (startX, startY, endX, endY)
        # Output: update class fields
        # empty check
        if len(centroids) == 0:
            # if empty, then no object detected in this frame, therefore,
			# mark all exisiting object as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    # if some object has been marked for enough times, deregister it
                    self.deregister(objectID)
            return self.objects
        
        # if Initial state: no object is being tracked, just register all
        if len(self.objects) == 0:
            for i in range(0, len(centroids)):
                self.register(centroids[i])
                
        # if Processing state: under tracking, try to match the input centroids to existing object centroids
        else:
			# get current objects and centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # find mininum distance among input centroid and tracked ones
            D = dist.cdist(np.array(objectCentroids), centroids)
            rows = D.min(axis=1).argsort() # sort index, ascending, col: current object in centroid list
            cols = D.argmin(axis=1)[rows] # row: matched previous object
            
            usedRows = set()
            usedCols = set()
			# To compare with previous state, loop over the combination of the (row, column) index
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = centroids[col] # update centroid
                self.disappeared[objectID] = 0 # reset disappearance count
				# keep trace of examined ones
                usedRows.add(row)
                usedCols.add(col)
            
            # NOT yet examined rows and columns.
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                # if previous object count (D.shape[0]) is more than current, check disappearance
				# iterate over the unused row indexes
                for row in unusedRows:
                    # unmatched previous objects in unusedRows
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
					# check if mark is above threshold
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                # if previous object count is smaller than current, check appearance.
                for col in unusedCols:
                    self.register(centroids[col])

        return self.objects

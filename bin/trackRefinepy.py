# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 19:03:45 2020

@author: CellDivision
"""

import pandas as pd
import numpy as np
import math
import re

def dist(x1,y1,x2,y2):
    return math.sqrt((float(x1)-float(x2))**2 + (float(y1)-float(y2))**2)

def lineage_search(tb,track_id,ln=[]):
    # The function walks from the beginning of certain lineage till the end based 
    # works for one parent-daughter relationship
    # Input: tb(relationship annotation table), track_id: track ID to search
    # Output: list
    parents = list((tb.loc[tb['track']==int(track_id),'predicted_parent']))
    daughter = list((tb.loc[tb['track']==int(track_id),'predicted_daughter']))
    if parents[0] is None:
        parents= [] 
    if daughter[0] is None:
        daughter=[]
    if len(parents)<=1:
      # if the track has no parents or one parent
      if len(daughter)==1:
        # if the track is the begin or in the middle of a lineage
        # daughter should not be mitosis
        if re.search("/", str(daughter[0])) is None:
          ln = lineage_search(tb,daughter[0],ln + [daughter[0]])

    return ln

# Author: Jeff Gui Yifan

#==============================================================================
#==================== Warning ! Unpublished script ============================
#==============================================================================


# The script does two things: find potential parent-daughter cells, and wipes out
# random false classification in the form of A-B-A
# To determine potential parent-daughter cells, appearance and disappearance time
# and location of the track are examined. Tracks appearing within certain 
# distance and time shift after another track's disappearance is considered as the daughter track.
# Parent-daughter track does not necessary mean mitosis event. In fact, it can
# be caused by either of the three events
# - 1. cells moving outside the view field and then come back, therefore assigned differently
# - 2. A mis-transition: Ilastik assign cells belong to the same lineage to two tracks
# - 3. Temporal loss of signal or segmentation issue
# - 4. Mitosis

def doTrackRefine(track):

  DIST_TOLERANCE = 90 # Distance to search for parent-daughter relatoinship
  div_trans_factor = 1.5 # recommanded
  FRAME_TOLERANCE = 5 # Time distance to search for parent-daughter relationship

  # Filter out false detection
  track = track.sort_values(by=['trackId','frame']) # sort by track ID


  #=================PART A: Relationship prediction=========================
  # annotation table: record appearance and disappearance information of the track
  track_count = len(np.unique(track['trackId']))
  ann = {"track" : [i for i in range(track_count)], 
        "app_frame": [0 for _ in range(track_count)],
        "disapp_frame" : [0 for _ in range(track_count)], 
        "app_x" :  [0 for _ in range(track_count)], # appearance coordinate
        "app_y" :  [0 for _ in range(track_count)],
        "disapp_x" :  [0 for _ in range(track_count)], # disappearance coordinate
        "disapp_y" :  [0 for _ in range(track_count)],
        "app_stage" :  [None for _ in range(track_count)], # cell cycle classification at appearance
        "disapp_stage" : [None for _ in range(track_count)], # cell cycle classification at disappearance
        "predicted_parent" : [None for _ in range(track_count)], # non-mitotic parent track TO-predict
        "predicted_daughter" : [None for _ in range(track_count)],
        "mitosis_parent" : [None for _ in range(track_count)], # mitotic parent track to predict
        "mitosis_daughter" : [None for _ in range(track_count)],
        "mitosis_identity" : [False for _ in range(track_count)]
  }
  
  short_tracks = []
  for i in range(track_count):
    cur_track = track[track['trackId']==i]
    # constraint A: track < 2 frame length tolerance is filtered out, No relationship can be deduced from that.
    ann['track'][i] = i
    # (dis-)appearance time
    ann['app_frame'][i] = min(cur_track['frame']) 
    ann['disapp_frame'][i] = max(cur_track['frame'])
    # (dis-)appearance coordinate
    ann['app_x'][i] = cur_track['x'].iloc[0]
    ann['app_y'][i] = cur_track['y'].iloc[0]
    ann['disapp_x'][i] = cur_track['x'].iloc[cur_track.shape[0]-1]
    ann['disapp_y'][i] = cur_track['y'].iloc[cur_track.shape[0]-1]
    if cur_track.shape[0] >= 2*FRAME_TOLERANCE:
      # record (dis-)appearance cell cycle classification, in time range equals to FRAME_TOLERANCE
      ann['app_stage'][i] = '-'.join(cur_track['predicted_class'].iloc[0:FRAME_TOLERANCE])
      ann['disapp_stage'][i] = '-'.join(cur_track['predicted_class'].iloc[(cur_track.shape[0]-FRAME_TOLERANCE): cur_track.shape[0]])
    else:
      ann['app_stage'][i] = cur_track['predicted_class'].iloc[0]
      ann['disapp_stage'][i] = cur_track['predicted_class'].iloc[cur_track.shape[0]-1]
      short_tracks.append(i)
  ann = pd.DataFrame(ann)
  track['lineageId'] = track['trackId'].copy() # erase original lineage ID, assign in following steps
  print("High quality tracks subjected to predict relationship: " + str(ann.shape[0] - len(short_tracks)))

  count = 0
  # Mitosis search 1
  #   Aim: to identify two appearing daughter tracks after one disappearing parent track
  #   Algorithm: find potential daughters, for each pair of them, 
  potential_daughter_pair_id = list(ann[list(map(lambda x:re.search('[MG1]', ann['app_stage'].iloc[x]) is not None and ann['track'].iloc[x] not in short_tracks, range(ann.shape[0])))]['track']) # daughter track must appear as M during mitosis
  for i in range(len(potential_daughter_pair_id)-1):
    for j in range(i+1, len(potential_daughter_pair_id)):
      # iterate over all pairs of potential daughters
      target_info_1 = ann[ann['track']==potential_daughter_pair_id[i]]
      target_info_2 = ann[ann['track']==potential_daughter_pair_id[j]]
      if target_info_1.shape[0]==0 or target_info_2.shape[1]==0: continue
      if dist(target_info_1['app_x'], target_info_1['app_y'], target_info_2['app_x'], target_info_2['app_y']) <= (DIST_TOLERANCE * div_trans_factor) and abs(int(target_info_1['app_frame']) - int(target_info_2['app_frame'])) < FRAME_TOLERANCE:
        # Constraint A: close distance
        # Constraint B: close appearing time
        
        # Find potential parent that disappear at M
        potential_parent = list(ann[list(map(lambda x:re.search('M', ann['disapp_stage'].iloc[x]) is not None and ann['mitosis_identity'].iloc[x]==False and x not in short_tracks, range(ann.shape[0])))]['track'])
        
        ann.loc[ann['track']==potential_daughter_pair_id[i],"mitosis_identity"] = "daughter"
        ann.loc[ann['track']==potential_daughter_pair_id[j],"mitosis_identity"] = "daughter"
        for k in range(len(potential_parent)):
          # spatial condition
          parent_x = int(ann[ann['track']==potential_parent[k]]["disapp_x"])
          parent_y = int(ann[ann['track']==potential_parent[k]]["disapp_y"])
          parent_disapp_time = int(ann[ann['track']==potential_parent[k]]["disapp_frame"])
          parent_id = int(ann[ann['track']==potential_parent[k]]["track"])
          if dist(target_info_1['app_x'], target_info_1['app_y'], parent_x, parent_y) <= DIST_TOLERANCE or dist(target_info_1['app_x'], target_info_2['app_y'], parent_x, parent_y) <= DIST_TOLERANCE:
              # Note, only one distance constaint met is accepted.
            # Constraint A: parent close to both daughter tracks' appearance
            if abs(int(target_info_1['app_frame']) - parent_disapp_time) < FRAME_TOLERANCE and abs(int(target_info_2['app_frame']) - parent_disapp_time) < FRAME_TOLERANCE:
                # Constraint B: parent disappearance time close to daughter's appearance
                  # update information in ann table
              ann.loc[ann['track']==int(target_info_1['track']),"mitosis_parent"] = parent_id
              ann.loc[ann['track']==int(target_info_2['track']), "mitosis_parent"] = parent_id
              ann.loc[ann['track']==parent_id,"mitosis_identity"] = "parent"
              ann.loc[ann['track']==parent_id,"mitosis_daughter"] = str(int(target_info_1['track'])) + "/" + str(int(target_info_2['track']))
                  # update information in track table
              track.loc[list(map(lambda x:track['trackId'].iloc[x]==int(target_info_1['track']) or 
                                track['trackId'].iloc[x]==int(target_info_2['track']), range(track.shape[0]))), ['lineageId','parentTrackId']] = parent_id
              count = count + 1

  print("Low confidence mitosis relations found:" + str(count))
  track = track.sort_values(by=['lineageId','trackId','frame'])

  count = 0
  # Mitosis search 2: 
  #   Aim: solve mitotic track (daughter) that appear near another mitotic track (parent).
  #   Algorithm: find the pool of tracks that appear as mitotic. For each, find nearby mitotic tracks.
  sub_ann = ann[ann['mitosis_identity'] != "daughter"]
  potential_daughter_trackId = list(sub_ann[list(map(lambda x:re.search('[MG1]', sub_ann['app_stage'].iloc[x]) is not None and sub_ann['track'].iloc[x] not in short_tracks, range(sub_ann.shape[0])))]['track']) # potential daughter tracks must appear at M phase during mitosis
  for i in range(len(potential_daughter_trackId)):
    target_info = ann[ann['track']==potential_daughter_trackId[i]]
    # extract all info in the frame when potential daughter appears
    searching_range = range(int(target_info['app_frame']) - FRAME_TOLERANCE, int(target_info['app_frame'])+1)
    searching = track[list(map(lambda x:track['frame'].iloc[x] in searching_range, range(track.shape[0])))]
    # search for M cells (potential parent)
    searching = searching[searching['predicted_class']=="M"]
    if searching.shape[0]==0: 
        continue 
    else:
        pot_ids = list(np.unique(searching['trackId']))
        for d in pot_ids:
            if d in short_tracks:
                pot_ids.remove(d)
        searching = track[list(map(lambda x:track['frame'].iloc[x] == int(target_info['app_frame']) and 
                                  track['trackId'].iloc[x] in pot_ids, range(track.shape[0])))]
        for j in range(searching.shape[0]):
          if dist(target_info['app_x'], target_info['app_y'], searching['x'].iloc[j], searching['y'].iloc[j]) <= DIST_TOLERANCE * div_trans_factor and int(searching['trackId'].iloc[j]) != int(target_info['track']):
            # Constraint A: close distance
            # Constraint B: non-self.
            if target_info['mitosis_parent'].iloc[0] is not None: 
              # if the potential daughter already has mitosis parent, will override.
              print("Warning: muiltiple mitosis parents found, only keep the last one.")
            
            ann.loc[ann['track']==int(potential_daughter_trackId[i]),"mitosis_parent"] = int(searching['trackId'].iloc[j])
            # label parent and daughter tracks as mitotic searched
            ann.loc[ann['track']==int(potential_daughter_trackId[i]),"mitosis_identity"] = "daughter"
            #print(paste("Daughter: ", potential_daughter_trackId[i], sep = ""))
            ann.loc[ann['track']==int(searching['trackId'].iloc[j]), "mitosis_identity"] = "parent"
            ann.loc[ann['track']==int(searching['trackId'].iloc[j]), "mitosis_daughter"] = potential_daughter_trackId[i]
            #print(paste("Parent: ", searching$trackId[j], sep = ""))
            # update lineage and parent track information of the daughter track
            track.loc[list(map(lambda x:track['trackId'].iloc[x]==int(target_info['track']), range(track.shape[0]))), ['lineageId','parentTrackId']] = int(searching['trackId'].iloc[j])
            count = count + 1

  print("High confidence mitosis relations found:" + str(count))
  track = track.sort_values(by=['lineageId','trackId','frame'])

  count = 0
  # Lineage search
  #   Aim: to correct tracks with exactly one parent/daughter (gap-filling)
  #   Algorithm: first record parents and daughters for each single track, then link them up by the relationship.
  #   Key function: lineage search, works on tracks with exactly one parent/daughter
      
  for i in range(ann.shape[0]):
    # vectors to store predicted parents & daughters
    parent = -1 # only keep one parent with the maximum length
    par_max_len = -1 # maximum length of parent 
    dau_max_len = -1 # only keep one daughter with maximum length
    daughters = -1
    # info of each iterated track
    cur_info = ann.iloc[i,]
    app_frame = int(cur_info['app_frame'])
    disapp_frame = int(cur_info['disapp_frame'])
    app_crd = (int(cur_info['app_x']), int(cur_info['app_y']))
    disapp_crd = (int(cur_info['disapp_x']), int(cur_info['disapp_y']))
    
    # if appear as S or disappear as G1, frame tolerance is amplified by 10 for parent or daughter range respectively
    adj_tol_par = FRAME_TOLERANCE
    adj_tol_dau = FRAME_TOLERANCE
    if re.search('S',cur_info['app_stage']) is not None:
      adj_tol_par = adj_tol_par * 5
    
    if re.search('G1', cur_info['disapp_stage']) is not None:
      adj_tol_dau = adj_tol_dau * 5

    parent_range = range(app_frame-adj_tol_par, app_frame)
    daughter_range = range(disapp_frame+1,disapp_frame+adj_tol_dau+1)
    # candidate parents and daughters drawn within frame tolerance range
    cdd_parent = ann[list(map(lambda x:ann['disapp_frame'].iloc[x] in parent_range, range(ann.shape[0])))]
    cdd_daughter = ann[list(map(lambda x:ann['app_frame'].iloc[x] in daughter_range, range(ann.shape[0])))]
    # verify parent relationship
    if cdd_parent.shape[0] > 0 and cur_info['mitosis_identity']!="daughter":
      
      for j in range(cdd_parent.shape[0]):
        cdd_crd = (int(cdd_parent['disapp_x'].iloc[j]), int(cdd_parent['disapp_y'].iloc[j]))
        if dist(app_crd[0],app_crd[1],cdd_crd[0],cdd_crd[1]) <= DIST_TOLERANCE and (int(cdd_parent['disapp_frame'].iloc[j]) - int(cdd_parent['app_frame'].iloc[j])) > par_max_len:
          # location constraint
          parent = int(cdd_parent['track'].iloc[j])
          par_max_len = int(cdd_parent['disapp_frame'].iloc[j]) - int(cdd_parent['app_frame'].iloc[j])
  
      if parent!=-1:
        ann.iloc[i,9] = parent
        count = count + 1

    # verify daughter relationship
    if cdd_daughter.shape[0] > 0 and cur_info['mitosis_identity']!="parent":
      for j in range(cdd_daughter.shape[0]):
        cdd_crd =  (int(cdd_daughter['app_x'].iloc[j]), int(cdd_daughter['app_y'].iloc[j]))
        if dist(disapp_crd[0],disapp_crd[1],cdd_crd[0],cdd_crd[1]) <= DIST_TOLERANCE and (int(cdd_daughter['disapp_frame'].iloc[j]) - int(cdd_daughter['app_frame'].iloc[j])) > dau_max_len:
          daughters = int(cdd_daughter['track'].iloc[j])
          dau_max_len = int(cdd_daughter['disapp_frame'].iloc[j]) - int(cdd_daughter['app_frame'].iloc[j])
  
      if daughters!=-1:
        ann.iloc[i, 10] = daughters
        count = count + 1
      
  print("Lineage relations found: " + str(count))
  # Based on predicted information, adjust track identity
  pool = [] # already searched track
  l = []
  for i in range(ann.shape[0]):
    if ann['predicted_parent'].iloc[i] is not None or ann['predicted_daughter'].iloc[i] is not None and ann['mitosis_parent'].iloc[i] is None:
      if not int(ann['track'].iloc[i]) in pool: 
        rlt = lineage_search(ann, int(ann['track'].iloc[i]), [int(ann['track'].iloc[i])])
        if len(rlt)>=2:
          l.append(list(rlt))
          pool = pool + rlt

  # for established lineage, assign parent track id to daughters
  for lineage in l:
    lineage_v = lineage
    for i in range(1,len(lineage_v)):
        track.loc[track['trackId']==lineage_v[i], ["trackId","lineageId"]] =lineage_v[0]

  track = track.sort_values(by=['lineageId','trackId','frame'])
  print("Lineage amount after reorganizing the lineage:" + str(len(np.unique(track['lineageId']))))

  return track

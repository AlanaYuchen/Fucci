#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:39:08 2020

@author: Full Moon
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_track(track, out, prefix):
    colors = {'G1':'red', 'S':'green', 'M':'gold', 'G2':'darkorange'}
    plt.figure(dpi=300,figsize=(8,8))
    for trk in np.unique(track['trackId']):
        trk_filtered = track[track['trackId']==trk]
        plt.plot(trk_filtered['mcy_intensity'], trk_filtered['gfp_intensity'], linestyle="dashed", color='lightgray', linewidth=1, zorder=1)
        plt.scatter(trk_filtered['mcy_intensity'], trk_filtered['gfp_intensity'], marker=".", s=5, c = trk_filtered['predicted_class'].map(colors), zorder=2)
    plt.title('Tracks', fontsize=24)
    plt.xlabel('mCherry', fontsize=14)
    plt.ylabel('GFP', fontsize=14)
    plt.savefig(out + prefix + 'plot.png')
    #plt.savefig('/Users/jefft/Desktop/plot.png')
    plt.show()
    return

def save_track(track, out, prefix):
    track.to_csv(out + prefix + 'track.csv', index=0)
    return

#=============================== Testing ======================================
'''
import pandas as pd
track = pd.read_csv('/Users/jefft/Desktop/BMI_Project/Fucci/data/test_out_cls_trk_rfd.csv')
plot_track(track, 'dummy', 'dummy')
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:39:08 2020

@author: Full Moon
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
'''
trackId = 2
dt_filtered = dt[dt['trackId']==trackId]
plt.scatter(dt_filtered['mcy_intensity'], dt_filtered['gfp_intensity'], marker=".", color='red')
plt.plot(dt_filtered['mcy_intensity'], dt_filtered['gfp_intensity'], linestyle="dashed", color='blue')
plt.title('Stage 1, track ' + str(trackId), fontsize=24)
plt.xlabel('mCherry', fontsize=14)
plt.ylabel('GFP', fontsize=14)
'''
'''
# bug, add frame label to each point
for i in range(dt_filtered.shape[0]-1):
    plt.text(x=dt_filtered['mcy_intensity'][i], y=dt_filtered['gfp_intensity'][i], s=str(dt_filtered['frame'][i]),
            fontsize=10, color = "r", verticalalignment='center', horizontalalignment='right')
'''
# plt.show()



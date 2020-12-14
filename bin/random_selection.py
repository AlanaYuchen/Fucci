#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:29:52 2020

@author: chenghui
"""
import numpy as np
import os
import skimage.io as io
import random

os.chdir('/Users/chenghui/Documents/BMI3/ICA/Fucci/data/')
frame=5
r=np.empty((10,frame))
for i in range(1,11):
    gfp_path = 'P'+i+'/P'+i+'_gfp.tif'
    mcy_path = 'P'+i+'/P'+i+'_mCherry.tif'
    gfp=io.imread(gfp_path)
    mcy=io.imread(mcy_path)
    chose=random.sample(range(0,289), frame)
    gfps=gfp[chose,:,:]
    mcys=mcy[chose,:,:]
    r[i-1,:]=chose
    io.imsave('P'+i+'/P'+i+'_gfps.tif',gfps)
    io.imsave('P'+i+'/P'+i+'_mcyss.tif',mcys)
    
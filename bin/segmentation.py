#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 22:08:13 2020

@author: Full Moon
"""
import numpy as np
def Sobel_segmentation(images):
    # input: uint8 / float8 image
    shape=images.shape
    sobel_result=np.zeros(shape, dtype='float64', order='C')
    Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    nrows = shape[1]
    ncols = shape[2]
    for m in range(images.shape[0]):
       image = images[m,:,:]
       mag = np.zeros((nrows, ncols))
       
       for i in range(nrows-2):
           for j in range(ncols-2):
               factor = image[i+1, j+1] + 1
               S1 = np.sum(np.multiply(Gx,image[i:i+3, j:j+3])) * 256 / factor
               S2 = np.sum(np.multiply(Gy,image[i:i+3, j:j+3])) * 256 / factor
               mag[i+1,j+1] = np.sqrt(pow(S1,2)+pow(S2,2))
       
       sobel_result[m,:,:] = mag
    return sobel_result

def doSeg(gfp_path, mcy_path):
    # Input: gfp and mCherry file path
    import skimage.io as io
    GFP = gfp_path
    MCY = mcy_path
    gfp=io.imread(GFP)
    mcy=io.imread(MCY)
    # Enhance contrast: histogram equalization
    import skimage.exposure as exposure
    gfp= np.asarray(list(map(lambda x: exposure.rescale_intensity(gfp[x,:,:], in_range=tuple(np.percentile(gfp[x,:,:], (2, 98)))),range(289))))
    mcy= np.asarray(list(map(lambda x: exposure.rescale_intensity(mcy[x,:,:], in_range=tuple(np.percentile(mcy[x,:,:], (2, 98)))), range(289))))
    rng = 3
    gfp=gfp[0:rng,:,:]
    mcy=mcy[0:rng,:,:]
    gfp=gfp.astype('uint32')
    mcy=mcy.astype('uint32')
    # take average intensity
    ave=np.add(gfp,mcy)/2/65535*255
    
    #Gaussian filter
    import skimage.filters as filters
    from skimage.filters import threshold_otsu
    ave_gau=np.asarray(list(map(lambda x: filters.gaussian(ave[x,:,:],sigma=2), range(rng))))

    # sobel kernal
    sobel_result = Sobel_segmentation(ave_gau)
    
    # global otsu
    otsu=np.asarray(list(map(lambda x: threshold_otsu(sobel_result[x,:,:]),range(rng))))
    global_otsu =np.asarray(list(map(lambda x: sobel_result[x,:,:]>=otsu[x],range(rng))))
    global_otsu = global_otsu.astype('uint8')
    
    # fill holes
    from skimage.morphology import reconstruction
    filled = np.empty(global_otsu.shape, dtype='uint8')
    #seeds = np.empty(global_otsu.shape, dtype='uint8')
    for i in range(global_otsu.shape[0]):    
        seed = np.copy(global_otsu[i,:,:])
        seed[1:-1, 1:-1] = global_otsu[i,:,:].max()
        #seeds[i,:,:] = seed
        filled[i,:,:] = reconstruction(seed, global_otsu[i,:,:], method='erosion')
    
    # watershed
    from skimage.morphology import watershed
    watershed_result=np.asarray(list(map(lambda x: watershed(filled[x,:,:],markers=None),range(rng))))
    
    # Output: mask, intensity rescaled gfp and mCherry image stacks
    return watershed_result, gfp, mcy
    
#============================= Testing ========================================
import os
os.chdir('/Users/chenghui/Documents/BMI3/ICA/Fucci')
gfp_path = 'data/P1_gfp.tif'
mcy_path = 'data/P1_mCherry.tif'
doSeg(gfp_path,mcy_path)




    
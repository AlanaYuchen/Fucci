#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:12:34 2020

@author: chenghui
"""

'''
    image_sobel <- sobel_operator(image_pre)
    image_sobel_rescaled <- rescaling(image_sobel)
    image_mask <- otsu_thresholding (image_sobel_rescaled)
    
    for i in range (number of frame of image_mask):
        mask_open[i] <- open_calculation(image_mask[i]) 
        mask_fillholes[i] <- filling_holes(image_open[i]) # filling closed holes
        mask_size_filtered[i] <- size_filter(mask_fillholes[i]) # remove small object
    
    # filling non-closed holes
    for i in range (number of frame of mask_size_filtered):
        label region in the mask, 
        for each region:
            get bounding_box, convex_hull_area, area, convex_image
            if convex_hull_area > 3* area:
               cell_mask <- binding_box(convex_image)
        cell_mask_erosed <- erosion (cell_mask) # reduce thickness of edges
    
retrurn cell_masked_erosed, gfp, mcy
'''

 # Sobel kernal and adaptive intensity stretching
    ave_gau = np.asarray(list(map(lambda x: filters.sobel(ave_gau[x,:,:]), range(rng))))
    ave_gau = np.asarray(list(map(lambda x: adaptive_rescaling(ave_gau[x,:,:], k=15), range(rng))))
    
    end_sobel = time.time()
    print("Finished edge detection: " + str(math.floor(end_sobel-start)) + " s.")

    # global otsu thresholding
    otsu = np.asarray(list(map(lambda x: threshold_otsu(ave_gau[x,:,:]),range(rng))))
    mask = np.asarray(list(map(lambda x: ave_gau[x,:,:]>=otsu[x],range(rng))))

    for i in range(mask.shape[0]):
        mask[i,:,:] = binary_opening(mask[i,:,:]) # open calculation, to denoise
        mask[i,:,:] = ndi.binary_fill_holes(mask[i,:,:]) # fill holes
        mask[i,:,:] = remove_small_objects(mask[i,:,:].astype('bool'), 1000) # size filter
    
    mask = mask.astype('uint8') * 255
    for i in range(mask.shape[0]):
        # inspect convex hull to fill non-connected ring edges
        props = measure.regionprops(measure.label(mask[i,:,:]), intensity_image=None)
    
        for k in range(len(props)):
            box = props[k].bbox
            if props[k].convex_area > 3 * props[k].area:
                mask[i,box[0]:box[2],box[1]:box[3]] = props[k].convex_image.copy() * 255
    
        # erosion, to exclude super thick edges
        mask[i,:,:] = binary_erosion(binary_erosion(mask[i,:,:]))
    
    mask =  mask.astype('uint8') * 255
    end = time.time()
    print("Finished generating mask: " + str(math.floor(end-start)) + " s.")
    # Output: mask, intensity rescaled gfp and mCherry image stacks
    #io.imsave('/Users/jefft/Desktop/mask_P1.tif',mask)
    return mask, gfp, mcy
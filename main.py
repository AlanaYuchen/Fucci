#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 14:47:45 2020

@author: Full Moon
"""
import warnings
warnings.filterwarnings("ignore") # ignore future warning in TensorFlow
import sys, getopt, os, re
basepath = os.path.abspath('./')
sys.path.append(basepath + '/bin')
import segmentation
import measureByMask
import cls_predict
import doTrack
import summary
import skimage.io as io

def main(argv):
   
    # I/O variables: output & input directory; GFP/mCherry/DIC file path and mode
    # The program supports two modes, 
    #   1) Single input: user specify single GFP/mCherry/DIC image datasets.
    #   2) Batch (directory input): user specify folder containing multiple datasets.
    out = ''
    ip = ''
    gfp_path = ''
    mCherry_path = ''
    dic_path = ''
    mode = ''
    
    # Internal variables: store I/O information, cnn model information, run switch.
    prefix = ''
    verbose = False
    gfp_list = []
    mcy_list = []
    dic_list = []
    cnn_path = basepath + '/data/model.h5'
    trh_F = 90 # distance tolerance (how Far neighbour)
    trh_T = 5 # time tolerance (how Time Far neighbour)
    
    # Collect and resolve user command
    try:
        opts, args = getopt.getopt(argv, "-h-i:-g:-m:-d:-o:-f:-t:-v", ["help","indir=","GFP_image=","mCherry_image=","DIC_image=", "outdir=", "threshold_F=", "threshold_T=", "verbose"])
    except getopt.GetoptError:
        print('fucci.py               -v <verbose> -h <help> \n ## Directory Mode ##  -i <input directory> -o <output directory> \n ##    File Mode   ##  -g <GFP image> -m <mCherry image> -d <DIC image> -o <output directory>')
        sys.exit()
    if  len(opts)==0:
        print('fucci.py               -v <verbose> -h <help> \n ## Directory Mode ##  -i <input directory> -o <output directory> \n ##    File Mode   ##  -g <GFP image> -m <mCherry image> -d <DIC image> -o <output directory>')
        sys.exit()
    for opt, arg in opts:
        if opt == '-h':
           print('fucci.py \n ## Directory Mode ##  -i <input directory> -o <output directory> \n ##    File Mode   ##  -g <GFP image> -m <mCherry image> -d <DIC image> -o <output directory>')
           sys.exit()
        elif opt in ("-i", "--indir"):
           ip = arg + "/"
           mode = 'dir'
        elif opt in ("-o", "--outdir"):
           out = arg + "/"
        elif opt in ("-g", "--GFP_image"):
           gfp_path = arg
           mode = 'file'
        elif opt in ("-m", "--mCherry_image"):
           mCherry_path = arg
        elif opt in ("-d", "--DIC_image"):
           dic_path = arg
        elif opt in ("-t", "--threshold_T"):
            trh_T = arg
        elif opt in ("-f", "--threshold_F"):
            trh_F = arg
        elif opt in ("-v", "--verbose"):
            verbose = True
    if len(ip) * len(gfp_path) or len(ip) * len(mCherry_path) or len(ip) * len(dic_path) != 0:
        # if two modes are used ambigiously
        print('Error! Use only one mode at a time.')
        sys.exit()
    
    if mode == 'dir':
        # in batch / directory mode, resolve file names in the directory
        if verbose: print("Resolving input directory.")
        for filename in os.listdir(ip):
            if re.match('(.*GFP.*).tif', filename):
                gfp_list.append(filename)
            elif re.match('(.*mCherry.*).tif', filename):
                mcy_list.append(filename)
            elif re.match('(.*DIC.*).tif', filename):
                dic_list.append(filename)
    else:
        # in single mode, extract file name from file path
        gfp_list.append(re.search('.*/(.*GFP.*)', gfp_path).group(1))
        mcy_list.append(re.search('.*/(.*mCherry.*)', mCherry_path).group(1))
        dic_list.append(re.search('.*/(.*DIC.*)', dic_path).group(1))
    
    gfp_list.sort()
    mcy_list.sort()
    dic_list.sort()
    
    # Wait for user to confirm file names.
    print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    print("GFP images:      " + str(gfp_list))
    print("mCherry images:  " + str(mcy_list))
    print("DIC images:      " + str(dic_list))
    print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
    
    if len(set([len(gfp_list), len(mcy_list), len(dic_list)])) != 1:
        # image file should have one-to-one relationship
        print("Error! Can not resolve datasets without same lengths.")
        sys.exit()
    go = input("\nImage files in correct order? [y/n] ")
    if go!='y':
        sys.exit()
    
    # Main functions
    for i in range(len(gfp_list)):
        if mode == 'dir':
            gfp_path = ip + gfp_list[i]
            mCherry_path = ip + mcy_list[i]
            dic_path = ip + dic_list[i]
        
        prefix = re.search('(.*)GFP.*', gfp_list[i]).group(1)
        
        # Step 1. Segmentation
        if verbose: print("\n>>> Segmentation\n")
        mask, gfp_pcd, mcy_pcd = segmentation.doSeg(gfp_path, mCherry_path)
        io.imsave('/Users/jefft/Desktop/P2_mask.tif', mask)

        # Step 2. Identify objects, retrieve resized images of each object
        if verbose: print("\n>>> Object Identification")
        obj_table, stacks = measureByMask.doMeasure(mask, gfp_pcd, mcy_pcd, dic_path)
        if verbose: print("Identified " + str(len(stacks)) + " objects. \n")
        
        # Step 3. Object classification
        if verbose: print(">>> Classification\n")
        obj_table = cls_predict.doPredict(obj_table, stacks, cnn_path)
        
        # Step 4. Tracking
        if verbose: print("\n>>> Tracking\n")
        tracks = doTrack.centroidTracking(obj_table, trh_F, trh_T)
        
        # Step 5. Output result
        summary.plot_track(tracks, out, prefix)
        summary.save_track(tracks, out, prefix)
    
    return
    
if __name__ == "__main__":
    main(sys.argv[1:])
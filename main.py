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

def main(argv):
    out = ''
    ip = ''
    gfp_path = ''
    mCherry_path = ''
    dic_path = ''
    mode = ''
    verbose = False
    gfp_list = []
    mcy_list = []
    dic_list = []
    cnn_path = basepath + '/data/model.h5'
    try:
        opts, args = getopt.getopt(argv, "hbi:g:m:d:o:", ["indir=","GFP_image=","mCherry_image=","DIC_image=", "outdir="])
        # h: switch-type parameter
        # i: / o: parameter must with some values
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
           gfp_path = arg + "/"
           mode = 'file'
        elif opt in ("-m", "--mCherry_image"):
           mCherry_path = arg + "/"
        elif opt in ("-d", "--DIC_image"):
           dic_path = arg + "/"
        elif opt in ("-v", "--verbose"):
            verbose = True
    if len(ip) * len(gfp_path) or len(ip) * len(mCherry_path) or len(ip) * len(dic_path) != 0:
        # if two modes are used ambigiously
        print('Error! Use only one mode at a time.')
        sys.exit()
    
    if mode == 'dir':
        if verbose: print("Resolving input directory.")
        for filename in os.listdir(ip):
            if re.match('(.*GFP.*).tif', filename):
                gfp_list.append(filename)
            elif re.match('(.*mCherry.*).tif', filename):
                mcy_list.append(filename)
            elif re.match('(.*DIC.*).tif', filename):
                dic_list.append(filename)
    else:
        gfp_list.append(re.search('.*/(.*GFP.*)', gfp_path).group(1))
        mcy_list.append(re.search('.*/(.*mCherry.*)', mCherry_path).group(1))
        dic_list.append(re.search('.*/(.*DIC.*)', dic_path).group(1))
        
    print("GFP images: " + str(gfp_list))
    print("mCherry images" + str(mcy_list))
    print("DIC images" + str(dic_list))
    if len(set(len(gfp_list), len(mcy_list), len(dic_list))) != 1:
        print("Error! Resolved datasets not of same length.")
        sys.exit()
    go = input("Image files in correct order? [y/n] ")
    if not go:
        sys.exit()
    
    for i in len(gfp_list):
        if mode == 'dir':
            gfp_path = ip + gfp_list[i]
            mCherry_path = ip + mcy_list[i]
            dic_path = ip + dic_list[i]
    
    # Step 1. Segmentation
    if verbose: print(">>> Segmentation\n")
    mask, gfp_pcd, mcy_pcd = doSeg(gfp_path, mCherry_path)
    
    # Step 2. Identify objects, retrieve resized images of each object
    if verbose: print(">>> Object Identification")
    obj_table, stacks = doMeasure(mask, gfp_pcd, mcy_pcd)
    if verbose: print("Identified " + str(len(stacks)) + " objects. \n")
    
    # Step 3. Object classification
    if verbose: print(">>> Classification\n")
    obj_table = cls_predict.doPredict(obj_table, stacks, cnn_path)
    
    # Step 4. Tracking
    if verbose: print(">>> Tracking\n")
    tracks = doTrack.centroidTracking(obj_table)
    
    # Step 5. Output result
    summary(tracks, out)
    
if __name__ == "__main__":
    main(sys.argv[1:])
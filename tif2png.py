#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:07:41 2020

@author: jefft

Convert tif file format to png.
"""

import os
from skimage import io
import re
import sys, getopt

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["indir=", "outdir="])
        # h: switch-type parameter
        # i: / o: parameter must with some values
    except getopt.GetoptError:
        print('tif2png.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
          if opt == '-h':
             print('tif2png.py -i <inputfile> -o <outputfile>')
             sys.exit()
          elif opt in ("-i", "--indir"):
             ip = arg + "/"
          elif opt in ("-o", "--outdir"):
             out = arg + "/"
    for filename in os.listdir(ip):
        if re.match('(.*).tif', filename):
            # if tif file
            img = io.imread(ip+filename)
            prefix = re.match('(.*).tif',filename).group(1)
            io.imsave(out + prefix + '.png', img)
        
if __name__ == "__main__":
    main(sys.argv[1:])
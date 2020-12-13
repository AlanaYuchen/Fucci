#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 09:04:18 2020

@author: jefft

Convert image mask format between json and tif/png
"""
from skimage import io
import sys, getopt, re, os, copy, time, json
from PIL import Image, ImageDraw
from skimage import measure

def main(argv):
    rev = False
    try:
        opts, args = getopt.getopt(argv, "hri:o:", ["indir=", "outdir="])
        # h: switch-type parameter, help
        # r: switch-type parameter, whether reverse conversion FROM json TO binary mask.
        # i: / o: parameter must with some values
    except getopt.GetoptError:
        print('mask2json.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
           print('mask2json.py -i <inputfile> -o <outputfile>')
           sys.exit()
        elif opt == '-r':
          rev = True
        elif opt in ("-i", "--indir"):
           ip = arg + "/"
        elif opt in ("-o", "--outdir"):
           out = arg + "/"
    
    if rev:
        jsonToMask(ip, out, 1200, 1200)
    else:
        maskToJson(ip, out)

def jsonToMask(ip, out, height, width):
    for filename in os.listdir(ip):
        if re.match('(.*).json', filename):
            # if json file
            with open(ip+filename,'r',encoding='utf8')as fp:
                img = Image.new('L',(height,width))
                j = json.load(fp)
                objs = j['metadata'] # containing all object areas
                draw = ImageDraw.Draw(img)
                for key in objs:
                    xys = objs[key]['xy'][1:] # first element is not coordinate
                    draw.polygon(xys, fill=255, outline=255)
                
            fname = j['file']['1']['fname']
            img.save(out + fname)
    return

def maskToJson(ip, out):
    with open('VIA_template.json','r', encoding='utf8') as fp:
        tmp = json.load(fp)
    obj_tmp = {'vid':'1', 'fig':0, 'z':[], 'xy':[], 'av':{}}
    for filename in os.listdir(ip):
        if re.match('(.*).png', filename) or re.match('(.*).tif', filename):
            # if tif file or png file
            out_dict = copy.deepcopy(tmp)
            #print('before' + str(len(out_dict['metadata'])))
            # metadata config
            out_dict['file']['1']['fname'] = filename
            out_dict['project']['created'] = int(time.time())
            # objects
            img = io.imread(ip+filename)
            contours = measure.find_contours(img, 0.5)
            for ct in contours:
                obj = copy.deepcopy(obj_tmp)
                edge = measure.approximate_polygon(ct, tolerance=0.45)
                for k in range(len(edge)):
                    edge[k] = [edge[k][1], edge[k][0]]
                obj['xy'] = [7] + list(edge.ravel()) # 7: unknown number found in template
                out_dict['metadata']['1_' + str(hash(str(ct)))] = obj
            if re.match('(.*).png', filename):
                prefix = re.match('(.*).png',filename).group(1)
            elif re.match('(.*).tif', filename):
                prefix = re.match('(.*).tif',filename).group(1)
            #print('after' + str(len(out_dict['metadata'])))
            with(open(out+prefix+'.json', 'w', encoding='utf8')) as fp:
                json.dump(out_dict,fp)
    return
        
if __name__ == "__main__":
    main(sys.argv[1:])

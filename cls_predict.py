#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 11:49:04 2020

@author: jefft
"""
class_dic = {'G1':0, 'S':1, 'G2':2, 'M':3}

import os
os.chdir('/Users/jefft/Desktop/BMI_Project/Fucci') # project path, absolute

model_path = 'data/model.h5' # relative directory of model
import tensorflow as tf
model = tf.keras.models.load_model(model_path)
import numpy as np

def load_image(d, resolve_class=False):
    # input: directory containing image files
    # resolve_class: for image labeled with cell cycle stage
    # output: shaped model input
    img_rows = 80
    img_cols = 80
    import os
    import skimage.io as io
    import re
    import keras.backend as bk
    import numpy as np
    imgs = []
    c = []
    for f in os.listdir(d):
        if re.search('.tif',f): # match tif files
            if resolve_class:
                c.append(re.search('\w*_\d+_(\w+).tif', f).group(1))
            img = io.imread(d+'/'+f)
            # reshape according to backend
            if bk.image_data_format() == "channels_first": # using tensorflow
                img = img.reshape(3, img_rows, img_cols)
            else:
                img = img.reshape(img_rows, img_cols, 3)
            # data type transform
            img = img.astype('float32')
            img /= 255
            imgs.append(img)
    if resolve_class:
        return np.stack(imgs, axis=0), c
    else:
        return np.stack(imgs, axis=0)

def class2stage(l):
    class2stage_dic = {'G1':0, 'S':1, 'G2':2, 'M':3}
    return list(map(lambda x: class2stage_dic[x], l))

def stage2class(l):
    stage2class_dic = {0:'G1', 1:'S', 2:'G2', 3:'M'}
    return list(map(lambda x: stage2class_dic[x], l))
    
    
s = load_image('data/training/valid', resolve_class=True)
prediction = model.predict(s[0])
prediction = stage2class(list(map(lambda x:np.where(prediction[x,:] == np.max(prediction[x,:]))[0][0], range(prediction.shape[0]))))
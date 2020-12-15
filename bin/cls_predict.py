#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 11:49:04 2020

@author: Full Moon
"""

#import os
#os.chdir(os.path.abspath("../")) # project path, absolute

# model_path = 'data/model.h5' # relative directory of model
import tensorflow as tf
import numpy as np
import skimage.io as io
import re
import keras.backend as bk
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # avoid problem on MacOS

def load_image_from_file(d, resolve_class=False):
    # The function reads in image files from a directory and reformat into model-readable format.
    # The purpose is to prepare trainning/validation/testing input, not for application.
    # input: directory containing image files
    # resolve_class: for image labeled with cell cycle stage
    # output: shaped model input
    img_rows = 80
    img_cols = 80
    import os
    
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

def load_image(imgs):
    # The function reformat imags from a list into model-readable format.
    img_rows, img_cols = imgs[0].shape[:2]
    for i in range(len(imgs)):
        if bk.image_data_format() == "channels_first": # using tensorflow
           imgs[i] = imgs[i].reshape(3, img_rows, img_cols)
        else:
           imgs[i] = imgs[i].reshape(img_rows, img_cols, 3)
    return np.stack(imgs, axis=0)

def class2stage(l):
    class2stage_dic = {'G1':0, 'S':1, 'G2':2, 'M':3}
    return list(map(lambda x: class2stage_dic[x], l))

def stage2class(l):
    stage2class_dic = {0:'G1', 1:'S', 2:'G2', 3:'M'}
    return list(map(lambda x: stage2class_dic[x], l))
    
def doPredict(obj_table, img_list, cnn_path):
    s = load_image(img_list)
    model = tf.keras.models.load_model(cnn_path)
    prediction = model.predict(s)
    prediction = stage2class(list(map(lambda x:np.where(prediction[x,:] == np.max(prediction[x,:]))[0][0], range(prediction.shape[0]))))
    # register prediction to the table
    obj_table['predicted_class'] = prediction
    return(obj_table)

#============================== Testing =======================================
'''
import os
os.chdir(os.path.abspath("../")) # project path, absolute
s = load_image('data/training/valid', resolve_class=True)
model = tf.keras.models.load_model('data/model.h5')
prediction = model.predict(s)
prediction = stage2class(list(map(lambda x:np.where(prediction[x,:] == np.max(prediction[x,:]))[0][0], range(prediction.shape[0]))))
'''



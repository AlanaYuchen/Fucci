#  -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:14:13 2020

@author: Full Moon
"""
 
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import os
import re
import skimage.io as io
import numpy as np
 
# project directory
os.chdir(os.path.abspath("../")) # project path, absolute

# relative directory to the project
train_dir = 'data/training/train'
valid_dir = 'data/training/valid'

# global variables
batch_size = 10
nb_classes = 4 # Cell cycle stage: G1/S/G2/M
class_dic = {'G1':0, 'S':1, 'G2':2, 'M':3}
epochs = 5
# input image dimensions
img_rows, img_cols = 80, 80
# number of convolutional filters
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

#load data
train_name = os.listdir(train_dir)
if '.DS_Store' in train_name:
    train_name.remove('.DS_Store')
label_train = list(map(lambda x: re.search('train_\d+_(\w+).tif', x).group(1), train_name))
label_train = list(map(lambda x: class_dic[x], label_train))

valid_name = os.listdir(valid_dir)
if '.DS_Store' in valid_name:
    valid_name.remove('.DS_Store')
label_valid = list(map(lambda x: re.search('valid_\d+_(\w+).tif', x).group(1), valid_name))
label_valid = list(map(lambda x: class_dic[x], label_valid))

train_set = np.empty((len(train_name), 80,80,3), dtype='uint8') # size:80*80, 3 channels
for f in range(len(train_name)):
    train_set[f,:,:,:] = io.imread(train_dir + '/' + train_name[f])
valid_set = np.empty((len(valid_name), 80,80,3), dtype='uint8')
for f in range(len(valid_name)):
    valid_set[f,:,:,:] = io.imread(valid_dir + '/' + valid_name[f])
 
# 根据不同的backend定下不同的格式
# input has three channels
if K.image_data_format() == "channels_first": # using tensorflow
    train_set = train_set.reshape(train_set.shape[0], 3, img_rows, img_cols)
    valid_set = valid_set.reshape(valid_set.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    train_set = train_set.reshape(train_set.shape[0], img_rows, img_cols, 3)
    valid_set = valid_set.reshape(valid_set.shape[0], img_rows, img_cols,3)
    input_shape = (img_rows, img_cols, 3)
 
# data type conversion
train_set = train_set.astype('float32')
valid_set = valid_set.astype('float32')
train_set /= 255
# X_test /= 255
valid_set /= 255
print('train_set shape:', train_set.shape)
print(train_set.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')
print(valid_set.shape[0], 'valid samples')
 
# 转换为one_hot类型
label_train = np_utils.to_categorical(label_train, nb_classes)

label_valid = np_utils.to_categorical(label_valid, nb_classes)
 
#building up model 
model = Sequential()
 
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='same',
                        input_shape=input_shape)) # convolutional kernal 1
model.add(Activation('relu')) #activation function
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]))) #convolutional kernal 2
model.add(Activation('relu')) #activation function
model.add(MaxPooling2D(pool_size=pool_size)) #pooling function
model.add(Dropout(0.25)) #neuron random inactivation
model.add(Flatten()) #flatten the data
model.add(Dense(128)) #fully-connected layer 1
model.add(Activation('relu')) #activation function
model.add(Dropout(0.5)) #random inactivation
model.add(Dense(nb_classes)) #fully-connected layer 2
model.add(Activation('softmax')) #Softmax layer
 
#compiling model
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
#training model
model.fit(train_set, label_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_set, label_valid))

# Save model
model.save('data/model.h5')

#evaluating model
#score = model.evaluate(X_test, Y_test, verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])
 


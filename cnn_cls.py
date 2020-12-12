#  -*- coding: utf-8 -*-

 
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
os.chdir('/Users/jefft/Desktop/BMI_Project/Fucci')

# relative directory to the project
train_dir = 'data/training/train'
valid_dir = 'data/training/valid'

# 全局变量
batch_size = 10
nb_classes = 4 # Cell cycle stage: G1/S/G2/M
class_dic = {'G1':0, 'S':1, 'G2':2, 'M':3}
epochs = 5
# input image dimensions
img_rows, img_cols = 80, 80
# 卷积滤波器的数量
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

#load data
train_name = os.listdir(train_dir)
if '.DS_Store' in train_name:
    train_name.remove('.DS_Store')
y_train = list(map(lambda x: re.search('train_\d+_(\w+).tif', x).group(1), train_name))
y_train = list(map(lambda x: class_dic[x], y_train))

valid_name = os.listdir(valid_dir)
if '.DS_Store' in valid_name:
    valid_name.remove('.DS_Store')
y_valid = list(map(lambda x: re.search('valid_\d+_(\w+).tif', x).group(1), valid_name))
y_valid = list(map(lambda x: class_dic[x], y_valid))

train_set = np.empty((len(train_name), 80,80,3), dtype='uint8')
for f in range(len(train_name)):
    train_set[f,:,:,:] = io.imread(train_dir + '/' + train_name[f])
valid_set = np.empty((len(valid_name), 80,80,3), dtype='uint8')
for f in range(len(valid_name)):
    valid_set[f,:,:,:] = io.imread(valid_dir + '/' + valid_name[f])
 
# 根据不同的backend定下不同的格式
# input has three channels
if K.image_data_format() == "channels_first": # using tensorflow
    train_set = train_set.reshape(train_set.shape[0], 3, img_rows, img_cols)
    # X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols) # no test set
    valid_set = valid_set.reshape(valid_set.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    train_set = train_set.reshape(train_set.shape[0], img_rows, img_cols, 3)
    # X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3) # no test set
    valid_set = valid_set.reshape(valid_set.shape[0], img_rows, img_cols,3)
    input_shape = (img_rows, img_cols, 3)
 
# 类型转换
train_set = train_set.astype('float32')
# X_test = X_test.astype('float32')
valid_set = valid_set.astype('float32')
train_set /= 255
# X_test /= 255
valid_set /= 255
print('train_set shape:', train_set.shape)
print(train_set.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')
print(valid_set.shape[0], 'valid samples')
 
# 转换为one_hot类型
y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)
y_valid = np_utils.to_categorical(y_valid, nb_classes)
 
#构建模型
model = Sequential()
 
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='same',
                        input_shape=input_shape)) # 卷积层1
model.add(Activation('relu')) #激活层
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]))) #卷积层2
model.add(Activation('relu')) #激活层
model.add(MaxPooling2D(pool_size=pool_size)) #池化层
model.add(Dropout(0.25)) #神经元随机失活
model.add(Flatten()) #拉成一维数据
model.add(Dense(128)) #全连接层1
model.add(Activation('relu')) #激活层
model.add(Dropout(0.5)) #随机失活
model.add(Dense(nb_classes)) #全连接层2
model.add(Activation('softmax')) #Softmax评分
 
#编译模型
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
#训练模型
model.fit(train_set, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_set, y_valid))

# Save model
model.save('data/model.h5')

#评估模型
#score = model.evaluate(X_test, Y_test, verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])
 


from __future__ import division, print_function

import numpy as np
import matplotlib
import os, json

from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from sklearn.preprocessing import OneHotEncoder

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image
from keras import applications
from keras.preprocessing.image import ImageDataGenerator

import bcolz
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

def vgg_preprocess(x):
    x = x - vgg_mean     # subtract mean
    #print(x)
    #print(x.shape)
    return x[:, ::-1]    # reverse axis bgr->rgb

def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical',
                target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

def get_data(path, target_size=(224,224)):
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    return np.concatenate([batches.next() for i in range(batches.samples)])  # Keras2

def onehot(x): return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())

#convolutional block definition
def ConvBlock(layers, model, filters):
    for i in range(layers): 
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(filters, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
#fully connected block definition
def FCBlock(model):
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    

#define the VGG model architecture
def VGG_16(weights_path=None):
    model = Sequential()
#   model.add(Lambda(vgg_preprocess, input_shape=(3,224,224))) #theano complient
    model.add(Lambda(vgg_preprocess, input_shape=(224,224,3))) #tensorflow complient

    ConvBlock(2, model, 64)
    ConvBlock(2, model, 128)
    ConvBlock(3, model, 256)
    ConvBlock(3, model, 512)
    ConvBlock(3, model, 512)

    model.add(Flatten())
    FCBlock(model)
    FCBlock(model)
    model.add(Dense(1000, activation='softmax'))
            
    if weights_path:
        model.load_weights(weights_path)
    
    return model


FILES_PATH = 'http://files.fast.ai/models/'; CLASS_FILE='imagenet_class_index.json'
fpath = get_file(CLASS_FILE, FILES_PATH+CLASS_FILE, cache_subdir='models')
fpath = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5','https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5', cache_subdir='models')

with open(fpath) as f: class_dict = json.load(f)
classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,3)) #tensorflow complient

#create the model like any python object
model = VGG_16(fpath)

path = "data/convbattle/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
    
batch_size=64
val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
batches = get_batches(path+'train', shuffle=False, batch_size=1)
"""
val_data = get_data(path+'valid')
trn_data = get_data(path+'train')
save_array(model_path+ 'train_data.bc', trn_data)
save_array(model_path + 'valid_data.bc', val_data)
"""
trn_data = load_array(model_path+'train_data.bc')
val_data = load_array(model_path+'valid_data.bc')

val_classes = val_batches.classes
trn_classes = batches.classes
val_labels = onehot(val_classes)
trn_labels = onehot(trn_classes)

trn_features = model.predict(trn_data, batch_size=batch_size)
val_features = model.predict(val_data, batch_size=batch_size)
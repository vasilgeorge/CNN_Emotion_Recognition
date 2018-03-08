from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform
from PIL import Image


def get_FER2013_data(num_training=28709, num_validation=4000, \
                     num_test=3589, subtract_mean=True):
    """
    Load the FER-2013 data
    """
    print("Begin loading FER data ... ")
    fer2013_dir = '/Users/vasilgeorge/Documents/Imperial Machine Learning MSc 2017:18/Machine Learning/assignment2_advanced/CNN_Emotion_Recognition/public'
    train_dir = fer2013_dir + '/Train/'
    test_dir = fer2013_dir + '/Test/'
    labels_name = fer2013_dir + '/labels_public.txt'

    labels = np.loadtxt(labels_name, skiprows=1,\
                        delimiter=',', usecols=1, dtype='int') #may need to be float

    X_train,X_val,X_test,y_train,y_val,y_test = [],[],[],[],[],[]
    for i in range(1,num_training+1):
        with Image.open(train_dir + str(i) + '.jpg').convert('L') as png:
            png_to_array = np.fromstring(png.tobytes(), dtype=np.uint8)
            png_to_array = png_to_array.reshape((1,png.size[1],png.size[0],1))
            X_train.append(png_to_array)
            y_train.append(labels[i-1])
        #    if i%100==0:
        #        print(np.round(i/30000,2) , '% loaded')

    for i in range(num_training+1,num_training+num_test+1):
        with Image.open(test_dir + str(i) + '.jpg').convert('L') as png:
            png_to_array = np.fromstring(png.tobytes(), dtype=np.uint8)
            png_to_array = png_to_array.reshape((1,png.size[1],png.size[0],1))
            X_test.append(png_to_array)
            y_test.append(labels[i-1])

    print("FER data loaded")

    X_train = np.concatenate(X_train)
    y_train = np.asarray(y_train)
    X_test = np.concatenate(X_test)
    y_test = np.asarray(y_test)

    # Subsample the data
    mask = list(range(num_training-num_validation, num_training))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training-num_validation))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train = np.subtract(X_train,mean_image,casting='unsafe')
        X_val = np.subtract(X_val,mean_image,casting='unsafe')
        X_test = np.subtract(X_test,mean_image,casting='unsafe')

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()


    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test}

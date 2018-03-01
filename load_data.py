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


def _parse_images_function():

        # Choose a random image from the training set
        images_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(
            "/Users/vasilgeorge/Documents/Imperial Machine Learning MSc 2017:18/Machine Learning/assignment2_advanced/CNN_Emotion_Recognition/public/Train/1.jpg"))
        image_reader=tf.WholeFileReader()
        _, image_file = image_reader.read(images_filename_queue)

        image = tf.image.decode_jpeg(image_file)


        # labels_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(
        #     "/Users/vasilgeorge/Documents/Imperial Machine Learning MSc 2017:18/Machine Learning/assignment2_advanced/CNN_Emotion_Recognition/public/labels_public.txt"))
        # labels_reader = tf.TextLineReader(skip_header_lines=1)
        # _, labels_file = labels_reader.read(labels_filename_queue)
        # label_file = tf.decode_raw(labels_file, tf.uint8)
        #label = labels_file[-1]


        image.set_shape((48, 48, 3))

        num_preprocess_threads = 1
        min_queue_examples = 256
        batch_size = 64
        image_batch = tf.train.shuffle_batch(
                                        [image],
                                        batch_size=batch_size,
                                        num_threads=num_preprocess_threads,
                                        capacity=min_queue_examples + 3 * batch_size,
                                        min_after_dequeue=min_queue_examples)

        with tf.Session() as sess:
            tf.local_variables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)
            image_tensors = sess.run(image)
            print(image_tensors)

            coord.request_stop()
            coord.join(threads)


def get_labels():

    filename = '/Users/vasilgeorge/Documents/Imperial Machine Learning MSc 2017:18/Machine Learning/assignment2_advanced/CNN_Emotion_Recognition/public/labels_public.txt'

    labels_file = open(filename, 'r')
    lines = labels_file.readlines()
    labels_length = len(lines)
    labels = np.array(labels_length)
    labels = ["" for x in range(labels_length)]
    i = 0
    j = 0
    for line in lines:
        if i != 0:
            line = line[:-1]
            labels[j] = line[-1]
            j += 1
        i += 1

    x = labels[0]
    i_x = int(x)
    print(i_x)

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
            if i%100==0:
                print(np.round(i/30000,2) , '% loaded')

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



dict1 = get_FER2013_data()
one_hot = []
values = dict1['y_train']
print(values)
for i in range(len(values)):
    one_hot.append( tf.one_hot(int(values[i]), 7))

with tf.Session() as sess:
    sess.run(one_hot[1])
    print(one_hot[1])

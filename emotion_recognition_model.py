from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


""" Input to this graph are going to be images of size 48x48 pixels which we are going to use to implement emotion recognition
    Our emotions are going to be: 0=Angry, 1=Fear, 2=Happy, 3=Sad, 4=Disgust, 5=Surprise, 6=Neutral"""

class Emotion_Recognition_Model(object, data_dict):

    def __init__(self):
        self.data_dict = data_dict



    """ In this function we are going to add all the necessary layers, require for the emotion recognition of the images"""
    def layers_graph(self):

        # Input Layer
        input_layer = tf.reshape (self.input_images, [-1, 48,48,1])
        # Output shape : [batch_size, 48, 48, 1] with regards to [batch_size, image_width, image_height, channels]

        # First convolutional layer - parameters need to be tuned
        conv1 = tf.layers.conv2d(
                                inputs = input_layer,
                                filters = 32,
                                kernel_size = [5,5],
                                padding = "same",
                                activation = tf.nn.relu
                                )

        # Output shape should be: [batch_size, 48, 48, 32 ]

        pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2,2], strides = 2)

        # Output shape should be: [batch_size, 24, 24, 32]

        conv2 = tf.layers.conv2d(
                                inputs = pool1,
                                filters = 64,
                                kernel_size = [5,5],
                                padding = "same",
                                activation = tf.nn.relu
                                )

        # Output shape should be: [batch_size, 24, 24, 64 ]

        pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2,2], strides = 2)

        # Output shape should be: [batch_size, 12, 12, 64]

        conv3 = tf.layers.conv2d(
                                inputs = pool1,
                                filters = 128,
                                kernel_size = [5,5],
                                padding = "same",
                                activation = tf.nn.relu
                                )

        # Output shape should be: [batch_size, 24, 24, 128 ]

        pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size = [2,2], strides = 2)

        # Output shape should be: [batch_size, 6, 6, 128]


        # Dense layer - a fully connected layer

        pool3_flat  = tf.reshape(pool3, [-1, 6 * 6 * 28])

        # pool2_flat shape : [1,9216] - [batch_size, features]

        dense = tf.layers.dense(inputs = pool3_flat, units = 1024, activation = tf.nn.relu)
        dropout = tf.nn.dropout(inputs = dense, rate = 0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Output shape : [batch_size, 1024(hidden units)]

        logits = tf.layers.dense(inputs = dropout, units = 7)

        # We define output units to be 7(seven), corresponding to each of our emotions
        # Output shape should be [batch_size, 7]


        predictions = {
            "classes" : tf.argmax(input = logits, axis = 1),
            "probabilities" : tf.nn.softmax(logits, name = 'softmax_tensor')
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss
        one_hot_labels = tf.one_hot(self.data_dict['y_train']











    def build_graph():


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


""" Input to this graph are going to be images of size 48x48 pixels which we are going to use to implement emotion recognition
    Our emotions are going to be: 0=Angry, 1=Fear, 2=Happy, 3=Sad, 4=Disgust, 5=Surprise, 6=Neutral"""

class Emotion_Recognition_Model(object):

    def __init__(self):
        self.input_images =



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



                        

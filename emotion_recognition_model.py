
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


""" Input to this graph are going to be images of size 48x48 pixels which we are going to use to implement emotion recognition
    Our emotions are going to be: 0=Angry, 1=Fear, 2=Happy, 3=Sad, 4=Disgust, 5=Surprise, 6=Neutral"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def _parse_function():
        filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(
            "/Users/vasilgeorge/Documents/Imperial Machine Learning MSc 2017:18/Machine Learning/assignment2_advanced/CNN_Emotion_Recognition/public/Train/1.jpg"))
        image_reader=tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)

        image = tf.image.decode_jpeg(image_file)

        with tf.Session() as sess:
            tf.local_variables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)

            image_tensor = sess.run([image])
            print(image_tensor)

            coord.request_stop()
            coord.join(threads)

_parse_function()

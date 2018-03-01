from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def _parse_function():

        # Choose a random image from the training set
        filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(
            "/Users/vasilgeorge/Documents/Imperial Machine Learning MSc 2017:18/Machine Learning/assignment2_advanced/CNN_Emotion_Recognition/public/Train/*.jpg"))
        image_reader=tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)

        image = tf.image.decode_jpeg(image_file)

        image.set_shape((48, 48, 3))

        num_preprocess_threads = 1
        min_queue_examples = 256
        batch_size = 64
        images = tf.train.shuffle_batch(
                                        [image],
                                        batch_size=batch_size,
                                        num_threads=num_preprocess_threads,
                                        capacity=min_queue_examples + 3 * batch_size,
                                        min_after_dequeue=min_queue_examples)

        with tf.Session() as sess:
            tf.local_variables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)
            image_tensors = sess.run(images)
            print(image_tensors)

            coord.request_stop()
            coord.join(threads)

_parse_function()

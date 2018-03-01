from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


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



get_labels()
